"""Capacity-aware routing shared by all calls in one service process.

No hedging, unbounded queues, hidden legacy fallback, or replay after output.
Counters are guarded by a threading lock so sync callers and multiple event
loops use the same capacity without binding asyncio primitives to a loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import openai

from .openai_chat import OpenAIChat, OpenAIChatResponse, LLMTokenBudgetExceeded
from .pool_config import PoolConfig, PoolSpec, ModelSpec, Member, ConfigError
from sirchmunk.utils import create_logger

logger = logging.getLogger(__name__)
CAPACITY_POLL_SECONDS = 0.025


class PoolError(RuntimeError):
    def __init__(self, code: str, status: int = 503):
        super().__init__(f"LLM routing: {code}")
        self.code, self.status = code, status


@dataclass
class ModelState:
    in_flight: int = 0
    started: int = 0
    succeeded: int = 0
    failed: int = 0
    canceled: int = 0
    cooldown_until: float = 0
    last_error: str | None = None


@dataclass
class RoutingGuard:
    error: str | None = None


class Scheduler:
    def __init__(self, clock=time.monotonic):
        self.lock = threading.RLock()
        self.clock = clock
        self.models: dict[str, ModelState] = defaultdict(ModelState)
        self.pools: dict[str, int] = defaultdict(int)
        self.waiting: dict[str, int] = defaultdict(int)
        self.weights: dict[tuple, dict[str, int]] = {}
        self.latest_revision = -1
        self.model_limits: dict[str, int] = {}
        self.pool_limits: dict[str, int] = {}

    def configure(self, config: PoolConfig):
        with self.lock:
            if config.revision > self.latest_revision:
                self.latest_revision = config.revision
                self.model_limits = {m.id: m.max_concurrency for m in config.models}
                self.pool_limits = {p.id: p.max_concurrency for p in config.pools}
                self.weights.clear()

    async def acquire(self, key, policy, models, members, excluded, revision, queue_deadline):
        queued = False
        try:
            while True:
                with self.lock:
                    if getattr(asyncio.current_task(), "cancelling", lambda: False)():
                        raise asyncio.CancelledError()
                    available = [m for m in members if m.model_id not in excluded and models[m.model_id].enabled]
                    if not available:
                        raise PoolError("no_eligible_model")
                    pool_limit = min(policy.max_concurrency, self.pool_limits.get(key, policy.max_concurrency))
                    eligible = [m for m in available
                                if self.models[m.model_id].in_flight < min(models[m.model_id].max_concurrency,
                                    self.model_limits.get(m.model_id, models[m.model_id].max_concurrency))
                                and self.models[m.model_id].cooldown_until <= self.clock()]
                    if self.pools[key] < pool_limit and eligible:
                        if policy.strategy == "weighted":
                            signature = (key, revision, tuple((m.model_id, m.weight) for m in eligible))
                            weights = self.weights.setdefault(signature, {})
                            # Bound signatures as eligibility changes under load.
                            if len(self.weights) > 512:
                                self.weights = {signature: weights}
                            for member in eligible:
                                weights[member.model_id] = weights.get(member.model_id, 0) + member.weight
                            chosen = max(eligible, key=lambda m: weights[m.model_id])
                            weights[chosen.model_id] -= sum(m.weight for m in eligible)
                        else:
                            chosen = min(eligible, key=lambda m: (m.priority, members.index(m)))
                        self.pools[key] += 1
                        state = self.models[chosen.model_id]
                        state.in_flight += 1
                        state.started += 1
                        return models[chosen.model_id]
                    if self.clock() >= queue_deadline:
                        raise PoolError("capacity_wait_timeout", 429)
                    if not queued:
                        if self.waiting[key] >= policy.max_queue_size:
                            raise PoolError("queue_full", 429)
                        self.waiting[key] += 1
                        queued = True
                await asyncio.sleep(min(CAPACITY_POLL_SECONDS, max(0, queue_deadline - self.clock())))
        finally:
            if queued:
                with self.lock:
                    self.waiting[key] -= 1

    def release(self, key, model_id, outcome, error=None, cooldown=0):
        with self.lock:
            self.pools[key] -= 1
            state = self.models[model_id]
            state.in_flight -= 1
            if outcome == "success":
                state.succeeded += 1
                state.last_error = None
                state.cooldown_until = 0
            elif outcome == "canceled":
                state.canceled += 1
            else:
                state.failed += 1
                state.last_error = error
                state.cooldown_until = max(state.cooldown_until, self.clock() + cooldown)

    def status(self, config):
        self.configure(config)
        with self.lock:
            return {
                "scope": "single_process", "revision": config.revision,
                "models": [{"id": m.id, "max_concurrency": m.max_concurrency,
                            **vars(self.models[m.id]),
                            "cooldown_seconds": round(max(0, self.models[m.id].cooldown_until - self.clock()), 2)}
                           for m in config.models],
                "pools": [{"id": p.id, "in_flight": self.pools[p.id], "waiting": self.waiting[p.id],
                           "max_concurrency": p.max_concurrency} for p in config.pools],
            }


def classify_error(exc):
    if isinstance(exc, (asyncio.TimeoutError, openai.APITimeoutError)):
        return "timeout", True
    if isinstance(exc, openai.APIConnectionError):
        return "connection_error", True
    if isinstance(exc, openai.APIStatusError):
        code = exc.status_code
        return f"upstream_http_{code}", code in (401, 403, 404, 408, 429) or code >= 500
    return "request_failed", False


class _Emission:
    def __init__(self, owner):
        self.owner = owner
        self.visible = False

    def info(self, text, **kwargs):
        if self.owner.guard.error:
            raise PoolError(self.owner.guard.error, 502)
        self.visible = self.visible or bool(text)
        self.owner._logger.info(text, **kwargs)

    async def ainfo(self, text, **kwargs):
        if self.owner.guard.error:
            raise PoolError(self.owner.guard.error, 502)
        self.visible = self.visible or bool(text)
        await self.owner._logger_async.info(text, **kwargs)


class _AsyncEmission:
    def __init__(self, emission):
        self.info = emission.ainfo


class RoutedChat(OpenAIChat):
    """OpenAIChat contract with request-local loggers and shared admission.

This adapter deliberately bypasses OpenAIChat's inner retry loop. Budget is
reserved/settled once per attempt; unknown failed-attempt usage is charged at
the reservation ceiling, not represented as a zero-token success.
"""
    def __init__(self, config: PoolConfig, scheduler: Scheduler, log_callback=None, client_factory=OpenAIChat, guard=None):
        config.validate_references()
        self.config = config.model_copy(deep=True)
        self.scheduler = scheduler
        scheduler.configure(config)
        self.client_factory = client_factory
        self.guard = guard or RoutingGuard()
        self._logger = create_logger(log_callback=log_callback, enable_async=False)
        self._logger_async = create_logger(log_callback=log_callback, enable_async=True)
        self.models = {m.id: m for m in config.models}
        target = config.active_target
        if target.kind == "pool":
            self.policy = next(p for p in config.pools if p.id == target.id)
            self.key = self.policy.id
        elif target.kind == "model":
            model = self.models[target.id]
            self.policy = PoolSpec(id="direct", name=model.name, strategy="failover",
                                   max_concurrency=model.max_concurrency, members=[Member(model_id=model.id)])
            self.key = "model:" + model.id
        else:
            raise ConfigError("Use the legacy client for legacy selection")
        self._model = f"{target.kind}:{target.id}"
        self.base_url = None

    async def achat(self, messages, stream=True, enable_thinking=False, **kwargs):
        if self.guard.error:
            raise PoolError(self.guard.error, 502)
        if "model" in kwargs:
            raise PoolError("model_override_not_allowed", 422)
        excluded = set()
        started = self.scheduler.clock()
        deadline = started + self.policy.request_timeout_seconds
        queue_deadline = min(deadline, started + self.policy.queue_timeout_seconds)
        last_code = "no_eligible_model"
        while len(excluded) < len(self.policy.members):
            if self.scheduler.clock() >= deadline:
                raise PoolError("route_timeout", 504)
            try:
                model = await self.scheduler.acquire(self.key, self.policy, self.models, self.policy.members,
                                                      excluded, self.config.revision, queue_deadline)
            except PoolError:
                if excluded:
                    raise PoolError(last_code, 502) from None
                raise
            excluded.add(model.id)
            outcome, code, cooldown = "failure", None, 0
            reservation, sent, response, client = 0, False, None, None
            emission = _Emission(self)
            try:
                if self.guard.error:
                    raise PoolError(self.guard.error, 502)
                call_kwargs = dict(kwargs)
                reservation = self._reserve_budget(messages, call_kwargs)
                remaining = min(model.timeout_seconds, max(0.001, deadline - self.scheduler.clock()))
                client = self.client_factory(api_key=model.key() or "local-no-auth", base_url=model.base_url,
                                             model=model.model, provider=model.provider, max_retries=0)
                client._logger = emission
                client._logger_async = _AsyncEmission(emission)
                call_kwargs["timeout"] = remaining
                request_kwargs = client._build_request_kwargs(stream, enable_thinking, **call_kwargs)
                sent = True
                response = await asyncio.wait_for(client._do_achat(messages, stream, request_kwargs), remaining)
                outcome = "success"
                if not response.model:
                    response.model = model.model
                response.usage = {**response.usage, "_model": response.model}
                return response
            except asyncio.CancelledError:
                outcome = "canceled"
                raise
            except LLMTokenBudgetExceeded:
                code = "token_budget_exceeded"
                raise
            except PoolError as exc:
                code = exc.code
                raise
            except Exception as exc:
                code, retryable = classify_error(exc)
                last_code = code
                cooldown = self.policy.failure_cooldown_seconds if retryable else 0
                if emission.visible:
                    self.guard.error = "stream_interrupted_no_replay"
                    raise PoolError("stream_interrupted_no_replay", 502) from None
                if not retryable:
                    raise PoolError(code, 422) from None
            finally:
                if reservation:
                    settled = response
                    reported = response.usage if response else {}
                    known = reported.get("total_tokens") or (
                        (reported.get("prompt_tokens") or 0) + (reported.get("completion_tokens") or 0))
                    if sent and not known:
                        settled = OpenAIChatResponse(content="", usage={"total_tokens": reservation})
                    self._settle_budget(reservation, settled)
                try:
                    if client is not None:
                        try:
                            await client._async_client.close()
                        finally:
                            client._client.close()
                except Exception:
                    logger.warning("LLM transport cleanup failed")
                finally:
                    self.scheduler.release(self.key, model.id, outcome, code, cooldown)
        raise PoolError(last_code, 502)

    def chat(self, messages, stream=True, enable_thinking=False, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.achat(messages, stream=stream, enable_thinking=enable_thinking, **kwargs))
        raise RuntimeError("Use await achat() inside an event loop")


_schedulers: dict[str, Scheduler] = {}
_scheduler_lock = threading.Lock()


def get_scheduler(path: str) -> Scheduler:
    with _scheduler_lock:
        if path not in _schedulers:
            _schedulers[path] = Scheduler()
        return _schedulers[path]

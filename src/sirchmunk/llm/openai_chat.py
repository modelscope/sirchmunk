# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import AsyncOpenAI, OpenAI

from sirchmunk.utils import LogCallback, create_logger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global LLM concurrency control
# ---------------------------------------------------------------------------
# Default max concurrent LLM calls across all OpenAIChat instances.
# Configurable via SIRCHMUNK_LLM_MAX_CONCURRENT environment variable.
_DEFAULT_LLM_MAX_CONCURRENT = 2

def _get_llm_max_concurrent() -> int:
    """Read max concurrent LLM calls from environment, with validation."""
    try:
        value = int(os.environ.get("SIRCHMUNK_LLM_MAX_CONCURRENT", _DEFAULT_LLM_MAX_CONCURRENT))
        return max(1, value)  # Ensure at least 1
    except (ValueError, TypeError):
        return _DEFAULT_LLM_MAX_CONCURRENT

# Global semaphore for async LLM calls. Python 3.10+ asyncio.Semaphore
# can be created without an event loop, so module-level creation is safe.
_llm_semaphore: asyncio.Semaphore = asyncio.Semaphore(_get_llm_max_concurrent())

# Transient error types that warrant automatic retry.
# - APIConnectionError / APITimeoutError: network-level failures.
# - InternalServerError: covers all 5xx (500, 502, 503, 504).
# - RateLimitError (429): provider throttling.
# - NotFoundError (404): many third-party OpenAI-compatible providers
#   return 404 during transient routing / load-balancer hiccups.
_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,   # all 5xx
    openai.RateLimitError,        # 429
    openai.NotFoundError,         # 404 — transient on some providers
)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0   # seconds
_DEFAULT_MAX_DELAY = 30.0   # seconds
_DEFAULT_CALL_TIMEOUT = 300.0  # seconds — per-attempt hard timeout


# ---------------------------------------------------------------------------
# Provider capability profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ProviderProfile:
    """Capability flags for a known LLM API provider.

    Attributes:
        name: Human-readable provider identifier.
        stream_options: Whether ``stream_options: {include_usage: true}`` is
            accepted.  Providers that reject unknown top-level params should
            set this to ``False``.
        thinking_param: ``extra_body`` key to toggle reasoning / thinking mode
            (e.g. ``"enable_thinking"``).  ``None`` when unsupported.
        thinking_content_field: Attribute name on the streaming ``delta`` (and
            non-streaming ``message``) that carries reasoning output.
            ``None`` when the provider does not expose thinking text.
    """
    name: str
    stream_options: bool = True
    thinking_param: Optional[str] = None
    thinking_content_field: Optional[str] = None


_PROVIDERS: Dict[str, _ProviderProfile] = {
    "openai":      _ProviderProfile("openai"),
    "azure":       _ProviderProfile("azure"),
    "deepseek":    _ProviderProfile("deepseek",    thinking_param="enable_thinking", thinking_content_field="reasoning_content"),
    "dashscope":   _ProviderProfile("dashscope",   thinking_param="enable_thinking", thinking_content_field="reasoning_content"),
    "siliconflow": _ProviderProfile("siliconflow",  thinking_param="enable_thinking", thinking_content_field="reasoning_content"),
    "volcengine":  _ProviderProfile("volcengine",   thinking_param="enable_thinking", thinking_content_field="reasoning_content"),
    "gemini":      _ProviderProfile("gemini",       thinking_param="reasoning_effort"),
    "zhipu":       _ProviderProfile("zhipu",        stream_options=False),
    "baichuan":    _ProviderProfile("baichuan",     stream_options=False),
    "moonshot":    _ProviderProfile("moonshot"),
    "mistral":     _ProviderProfile("mistral"),
    "yi":          _ProviderProfile("yi"),
    "together":    _ProviderProfile("together"),
    "groq":        _ProviderProfile("groq"),
    "cohere":      _ProviderProfile("cohere"),
    "minimax":     _ProviderProfile("minimax"),
}

# URL substring → provider name.  More specific patterns must precede less
# specific ones (e.g. ``openai.azure.com`` before ``openai.com``).
_URL_PATTERNS = [
    ("openai.azure.com",  "azure"),
    ("openai.com",        "openai"),
    ("deepseek.com",      "deepseek"),
    ("dashscope",         "dashscope"),
    ("bigmodel.cn",       "zhipu"),
    ("moonshot.cn",       "moonshot"),
    ("mistral.ai",        "mistral"),
    ("googleapis.com",    "gemini"),
    ("lingyiwanwu.com",   "yi"),
    ("01.ai",             "yi"),
    ("siliconflow",       "siliconflow"),
    ("volces.com",        "volcengine"),
    ("together.xyz",      "together"),
    ("groq.com",          "groq"),
    ("cohere",            "cohere"),
    ("baichuan-ai.com",   "baichuan"),
    ("minimax",           "minimax"),
]

_DEFAULT_PROFILE = _ProviderProfile("generic")


@dataclass
class OpenAIChatResponse:
    """Structured response from an OpenAI-compatible chat completion API."""

    content: str
    role: str = "assistant"
    usage: Dict[str, int] = field(default_factory=dict)
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    logprobs: Any = None
    thinking_content: Optional[str] = None

    def __str__(self) -> str:
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary."""
        result: Dict[str, Any] = {
            "content": self.content,
            "role": self.role,
            "usage": self.usage,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "logprobs": self.logprobs,
        }
        if self.thinking_content:
            result["thinking_content"] = self.thinking_content
        return result


# ---------------------------------------------------------------------------
# Stream accumulator — deduplicates logic between sync / async paths
# ---------------------------------------------------------------------------

@dataclass
class _StreamAccumulator:
    """Mutable accumulator updated chunk-by-chunk during streaming.

    Using ``list`` + ``join`` for O(n) string assembly instead of
    repeated ``+=`` concatenation which is O(n²).
    """
    content_parts: List[str] = field(default_factory=list)
    thinking_parts: List[str] = field(default_factory=list)
    role: str = "assistant"
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    model: str = ""

    def to_response(self) -> OpenAIChatResponse:
        """Build the final immutable response from accumulated state."""
        return OpenAIChatResponse(
            content="".join(self.content_parts),
            role=self.role,
            usage=self.usage,
            model=self.model,
            finish_reason=self.finish_reason,
            thinking_content="".join(self.thinking_parts) or None,
        )


class OpenAIChat:
    """Unified client for OpenAI-compatible chat completion APIs.

    Supports all major LLM providers (OpenAI, DeepSeek, DashScope/Qwen,
    Gemini, Zhipu, Moonshot, Mistral, etc.) through a single interface.
    Provider-specific capabilities (thinking mode, stream options) are
    automatically negotiated via :class:`_ProviderProfile`.
    """

    def __init__(
            self,
            api_key: str = None,
            base_url: str = None,
            model: str = None,
            log_callback: LogCallback = None,
            max_retries: int = _DEFAULT_MAX_RETRIES,
            retry_base_delay: float = _DEFAULT_BASE_DELAY,
            retry_max_delay: float = _DEFAULT_MAX_DELAY,
            call_timeout: float = _DEFAULT_CALL_TIMEOUT,
            provider: Optional[str] = None,
            **kwargs,
    ):
        """Initialize the chat client.

        Args:
            api_key: API key for the LLM provider.
            base_url: Base URL for the API endpoint (OpenAI-compatible).
            model: Model identifier for chat completions.
            log_callback: Optional callback for streaming log output.
            max_retries: Maximum retries for transient API errors.
            retry_base_delay: Initial backoff delay in seconds (doubled each retry).
            retry_max_delay: Upper bound on backoff delay in seconds.
            call_timeout: Per-attempt hard timeout in seconds.  Applies to both
                sync and async calls.  ``0`` or negative disables the timeout.
            provider: Explicit provider name (e.g. ``"deepseek"``, ``"dashscope"``).
                Overrides automatic URL-based detection.  Useful when the
                ``base_url`` points to a proxy / gateway that hides the real
                provider.
            **kwargs: Extra keyword arguments forwarded to the API ``create`` call.
        """
        self.base_url = base_url
        _timeout_kw = {"timeout": call_timeout} if call_timeout and call_timeout > 0 else {}
        self._client = OpenAI(api_key=api_key, base_url=base_url, **_timeout_kw)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **_timeout_kw)

        self._model = model
        self._kwargs = kwargs
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._call_timeout = call_timeout if call_timeout and call_timeout > 0 else None

        if provider:
            self._provider = _PROVIDERS.get(provider, _DEFAULT_PROFILE)
        else:
            self._provider = self._detect_provider(base_url)

        self._logger = create_logger(log_callback=log_callback, enable_async=False)
        self._logger_async = create_logger(log_callback=log_callback, enable_async=True)

    # ------------------------------------------------------------------
    # Provider detection & request building
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_provider(base_url: str) -> _ProviderProfile:
        """Detect the LLM provider from *base_url* and return its profile.

        Falls back to ``_DEFAULT_PROFILE`` (safe generic defaults) when the
        URL does not match any known provider.
        """
        url = (base_url or "").lower()
        for pattern, name in _URL_PATTERNS:
            if pattern in url:
                return _PROVIDERS.get(name, _DEFAULT_PROFILE)
        return _DEFAULT_PROFILE

    def _build_request_kwargs(
            self,
            stream: bool,
            enable_thinking: Optional[bool],
            **kwargs,
    ) -> Dict[str, Any]:
        """Merge instance-level and call-level kwargs for the API request.

        Precedence (highest wins): call-level kwargs > instance-level ``_kwargs``.
        ``extra_body`` dicts are deep-merged so fields from both levels coexist.

        Thinking parameters are **only** injected when the detected provider
        profile declares support (``thinking_param is not None``), preventing
        ``400 Bad Request`` errors on providers that reject unknown fields.
        """
        request_kwargs = {**self._kwargs, **kwargs}
        profile = self._provider

        extra_body = {
            **(self._kwargs.get("extra_body") or {}),
            **(kwargs.get("extra_body") or {}),
        }

        if enable_thinking is not None and profile.thinking_param:
            if profile.thinking_param == "reasoning_effort":
                if enable_thinking:
                    extra_body["reasoning_effort"] = "high"
                else:
                    extra_body.pop("reasoning_effort", None)
            else:
                extra_body[profile.thinking_param] = enable_thinking

        if extra_body:
            request_kwargs["extra_body"] = extra_body

        if stream and profile.stream_options and "stream_options" not in request_kwargs:
            request_kwargs["stream_options"] = {"include_usage": True}

        return request_kwargs

    # ------------------------------------------------------------------
    # Response parsing helpers (shared by sync & async paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_model_dump(usage_obj: Any) -> Dict[str, int]:
        """Convert a usage object to a plain dict.

        Handles Pydantic v2 models (``.model_dump()``), Pydantic v1
        (``.dict()``), raw dicts, and arbitrary objects gracefully.
        """
        if usage_obj is None:
            return {}
        if isinstance(usage_obj, dict):
            return usage_obj
        for method in ("model_dump", "dict"):
            fn = getattr(usage_obj, method, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        return vars(usage_obj) if hasattr(usage_obj, "__dict__") else {}

    def _process_stream_chunk(
            self,
            chunk: Any,
            acc: _StreamAccumulator,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Update *acc* with a single stream chunk.

        Returns ``(role_delta, content_delta, thinking_delta)`` — any may
        be ``None``.  The caller is responsible for logging these deltas.
        """
        if chunk.usage:
            acc.usage = self._safe_model_dump(chunk.usage)
        if chunk.model:
            acc.model = chunk.model
        if not chunk.choices:
            return None, None, None

        delta = chunk.choices[0].delta
        role_delta: Optional[str] = None
        content_delta: Optional[str] = None
        thinking_delta: Optional[str] = None

        if delta.role:
            acc.role = delta.role
            role_delta = delta.role

        thinking_field = self._provider.thinking_content_field
        if thinking_field:
            tc = getattr(delta, thinking_field, None)
            if tc:
                acc.thinking_parts.append(tc)
                thinking_delta = tc

        if delta.content:
            acc.content_parts.append(delta.content)
            content_delta = delta.content

        if chunk.choices[0].finish_reason:
            acc.finish_reason = chunk.choices[0].finish_reason

        return role_delta, content_delta, thinking_delta

    def _parse_non_stream_response(self, resp: Any) -> OpenAIChatResponse:
        """Parse a complete (non-streaming) API response into a response object.

        Includes a guard against empty ``choices`` which some providers
        return on malformed requests or internal errors.
        """
        usage = self._safe_model_dump(resp.usage)

        if not resp.choices:
            logger.warning("[LLM] API returned empty choices list")
            return OpenAIChatResponse(
                content="",
                model=resp.model or self._model,
                usage=usage,
            )

        message = resp.choices[0].message
        thinking_field = self._provider.thinking_content_field
        thinking = ""
        if thinking_field:
            thinking = getattr(message, thinking_field, "") or ""

        return OpenAIChatResponse(
            content=message.content or "",
            role=message.role,
            usage=usage,
            model=resp.model or self._model,
            finish_reason=resp.choices[0].finish_reason,
            thinking_content=thinking or None,
        )

    # ------------------------------------------------------------------
    # Retry infrastructure
    # ------------------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff with jitter for retry *attempt* (0-based)."""
        delay = min(self._retry_base_delay * (2 ** attempt), self._retry_max_delay)
        return delay * (0.5 + random.random() * 0.5)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True if *exc* is a transient error worth retrying."""
        return isinstance(exc, _RETRYABLE_ERRORS)

    def chat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool = True,
            enable_thinking: Optional[bool] = None,
            **kwargs,
    ) -> OpenAIChatResponse:
        """
        Generate a chat completion synchronously.

        Automatically retries on transient API errors (404 from provider
        routing hiccups, 429 rate-limits, 5xx server errors, and connection
        failures) with exponential backoff.

        Args:
            messages (List[Dict[str, Any]]): A list of messages for the chat.
            stream (bool): Whether to stream the response.
            enable_thinking (Optional[bool]): Whether to enable model thinking/reasoning.
                Sent via ``extra_body``. Defaults to None (uses instance-level setting
                from ``extra_body`` if provided, otherwise disabled).
            **kwargs: Additional keyword arguments merged with instance-level kwargs
                and forwarded to the OpenAI API. Call-level kwargs take precedence.

        Returns:
            OpenAIChatResponse: The structured response containing content, usage, etc.
        """
        request_kwargs = self._build_request_kwargs(stream, enable_thinking, **kwargs)
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                return self._do_chat(messages, stream, request_kwargs)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc) or attempt >= self._max_retries:
                    raise
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "[LLM] chat() attempt %d/%d failed (%s: %s), retrying in %.1fs",
                    attempt + 1, self._max_retries + 1,
                    type(exc).__name__, exc, delay,
                )
                time.sleep(delay)

        raise last_exc  # unreachable, but keeps type checkers happy

    def _do_chat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool,
            request_kwargs: Dict[str, Any],
    ) -> OpenAIChatResponse:
        """Single-attempt synchronous chat completion (no retry)."""
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, stream=stream, **request_kwargs
        )

        if not stream:
            result = self._parse_non_stream_response(resp)
            self._logger.info(f"[role={result.role}] {result.content}")
            return result

        acc = _StreamAccumulator(model=self._model)
        in_thinking = False
        for chunk in resp:
            role_delta, content_delta, thinking_delta = self._process_stream_chunk(chunk, acc)
            if role_delta:
                self._logger.info(f"[role={role_delta}] ", end="", flush=True)
            if thinking_delta:
                if not in_thinking:
                    self._logger.info("<think>", end="", flush=True)
                    in_thinking = True
                self._logger.info(thinking_delta, end="", flush=True)
            if content_delta:
                if in_thinking:
                    self._logger.info("</think>\n", end="", flush=True)
                    in_thinking = False
                self._logger.info(content_delta, end="", flush=True)
        if in_thinking:
            self._logger.info("</think>\n", end="", flush=True)
        self._logger.info("", end="\n", flush=True)

        return acc.to_response()

    async def achat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool = True,
            enable_thinking: Optional[bool] = None,
            **kwargs,
    ) -> OpenAIChatResponse:
        """
        Generate a chat completion asynchronously.

        Automatically retries on transient API errors with exponential
        backoff (same policy as :meth:`chat`).

        Args:
            messages (List[Dict[str, Any]]): A list of messages for the chat.
            stream (bool): Whether to stream the response.
            enable_thinking (Optional[bool]): Whether to enable model thinking/reasoning.
                Sent via ``extra_body``. Defaults to None (uses instance-level setting
                from ``extra_body`` if provided, otherwise disabled).
            **kwargs: Additional keyword arguments merged with instance-level kwargs
                and forwarded to the OpenAI API. Call-level kwargs take precedence.

        Returns:
            OpenAIChatResponse: The structured response containing content, usage, etc.
        """
        request_kwargs = self._build_request_kwargs(stream, enable_thinking, **kwargs)
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                coro = self._do_achat(messages, stream, request_kwargs)
                if self._call_timeout:
                    return await asyncio.wait_for(coro, timeout=self._call_timeout)
                return await coro
            except asyncio.TimeoutError:
                last_exc = asyncio.TimeoutError(
                    f"LLM call timed out after {self._call_timeout}s "
                    f"(attempt {attempt + 1}/{self._max_retries + 1})"
                )
                if attempt >= self._max_retries:
                    raise last_exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "[LLM] achat() attempt %d/%d timed out after %.0fs, retrying in %.1fs",
                    attempt + 1, self._max_retries + 1, self._call_timeout, delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc) or attempt >= self._max_retries:
                    raise
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "[LLM] achat() attempt %d/%d failed (%s: %s), retrying in %.1fs",
                    attempt + 1, self._max_retries + 1,
                    type(exc).__name__, exc, delay,
                )
                await asyncio.sleep(delay)

        raise last_exc  # unreachable, but keeps type checkers happy

    async def _do_achat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool,
            request_kwargs: Dict[str, Any],
    ) -> OpenAIChatResponse:
        """Single-attempt asynchronous chat completion (no retry).

        Uses global semaphore to limit concurrent LLM calls across all
        OpenAIChat instances, preventing API throttling cascades.
        """
        # Log when we need to wait for a semaphore slot
        if _llm_semaphore.locked():
            logger.debug("[LLM] Waiting for semaphore slot (max_concurrent=%d)...",
                         _get_llm_max_concurrent())

        async with _llm_semaphore:
            resp = await self._async_client.chat.completions.create(
                model=self._model, messages=messages, stream=stream, **request_kwargs
            )

            if not stream:
                result = self._parse_non_stream_response(resp)
                await self._logger_async.info(f"[role={result.role}] {result.content}")
                return result

            acc = _StreamAccumulator(model=self._model)
            in_thinking = False
            async for chunk in resp:
                role_delta, content_delta, thinking_delta = self._process_stream_chunk(chunk, acc)
                if role_delta:
                    await self._logger_async.info(f"[role={role_delta}] ", end="", flush=True)
                if thinking_delta:
                    if not in_thinking:
                        await self._logger_async.info("<think>", end="", flush=True)
                        in_thinking = True
                    await self._logger_async.info(thinking_delta, end="", flush=True)
                if content_delta:
                    if in_thinking:
                        await self._logger_async.info("</think>\n", end="", flush=True)
                        in_thinking = False
                    await self._logger_async.info(content_delta, end="", flush=True)
            if in_thinking:
                await self._logger_async.info("</think>\n", end="", flush=True)
            await self._logger_async.info("", end="\n", flush=True)

            return acc.to_response()

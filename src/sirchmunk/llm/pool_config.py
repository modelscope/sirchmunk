"""Private, versioned LLM configuration. No provider calls occur in this module.

Single-service-process storage: serialize writers, fsync then atomically replace.
The file belongs in the private work volume, never in static resources or Git.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

CONFIG_FILENAME = "llm-pools.json"
MAX_MODELS = 128
MAX_POOLS = 64


class ConfigError(RuntimeError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ModelSpec(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    name: str = Field(min_length=1, max_length=100)
    base_url: str = Field(max_length=2048)
    model: str = Field(min_length=1, max_length=200)
    provider: str | None = Field(default=None, max_length=40)
    enabled: bool = True
    max_concurrency: int = Field(default=4, ge=1, le=256, strict=True)
    timeout_seconds: float = Field(default=120, ge=1, le=600, allow_inf_nan=False)
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value):
        try:
            parsed = urlsplit(value)
            valid = parsed.scheme in ("http", "https") and parsed.hostname and parsed.port != 0
            if not valid or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError()
        except ValueError:
            raise ValueError("Use an http(s) API base URL without credentials, query or fragment") from None
        return value.rstrip("/")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value):
        if value:
            from .openai_chat import _PROVIDERS
            if value not in _PROVIDERS:
                raise ValueError("Unknown provider profile")
        return value or None

    def key(self) -> str:
        return os.getenv(self.api_key_env, "") if self.api_key_env else self.api_key.get_secret_value()


class Member(StrictModel):
    model_id: str
    weight: int = Field(default=1, ge=1, le=10000, strict=True)
    priority: int = Field(default=0, ge=0, le=10000, strict=True)


class PoolSpec(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    name: str = Field(min_length=1, max_length=100)
    enabled: bool = True
    strategy: Literal["weighted", "failover"] = "weighted"
    max_concurrency: int = Field(default=8, ge=1, le=1024, strict=True)
    queue_timeout_seconds: float = Field(default=15, ge=0, le=120, allow_inf_nan=False)
    max_queue_size: int = Field(default=64, ge=0, le=4096, strict=True)
    request_timeout_seconds: float = Field(default=180, ge=1, le=1800, allow_inf_nan=False)
    failure_cooldown_seconds: float = Field(default=15, ge=0, le=300, allow_inf_nan=False)
    members: list[Member] = Field(default_factory=list, max_length=MAX_MODELS)


class Target(StrictModel):
    kind: Literal["legacy", "model", "pool"] = "legacy"
    id: str | None = None


class PoolConfig(StrictModel):
    schema_version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    models: list[ModelSpec] = Field(default_factory=list, max_length=MAX_MODELS)
    pools: list[PoolSpec] = Field(default_factory=list, max_length=MAX_POOLS)
    active_target: Target = Field(default_factory=Target)

    def validate_references(self):
        models = {m.id: m for m in self.models}
        pools = {p.id: p for p in self.pools}
        if len(models) != len(self.models) or len(pools) != len(self.pools):
            raise ConfigError("Duplicate model or pool ID")
        for pool in self.pools:
            ids = [m.model_id for m in pool.members]
            if len(set(ids)) != len(ids) or any(i not in models for i in ids):
                raise ConfigError("Pool has duplicate or missing model references")
        target = self.active_target
        if target.kind == "legacy":
            if target.id is not None:
                raise ConfigError("Legacy selection has no ID")
        elif target.kind == "model":
            if target.id not in models or not models[target.id].enabled:
                raise ConfigError("Selected model is missing or disabled", 409)
        else:
            pool = pools.get(target.id)
            if not pool or not pool.enabled or not any(models[m.model_id].enabled for m in pool.members):
                raise ConfigError("Selected pool needs at least one enabled model", 409)

    def public(self) -> dict:
        data = self.model_dump(mode="json", exclude={"models": {"__all__": {"api_key"}}})
        for item, model in zip(data["models"], self.models):
            item["has_api_key"] = bool(model.key())
        data["legacy"] = {
            "name": "旧单模型配置", "model": os.getenv("LLM_MODEL_NAME", "gpt-5.2"),
            "configured": bool(os.getenv("LLM_API_KEY", "")),
        }
        return data


def _parse(cls, data):
    try:
        return cls.model_validate(data)
    except ValidationError as exc:
        # Never expose Pydantic input/ctx: those may contain supplied secrets.
        fields = ", ".join(".".join(map(str, e["loc"])) for e in exc.errors())
        raise ConfigError(f"Invalid configuration fields: {fields}") from None


class ConfigStore:
    def __init__(self, work_path: str | Path):
        from sirchmunk.storage.lifecycle import claim_work_root, get_runtime
        claim_work_root(work_path)
        self.path = Path(work_path).expanduser().resolve() / CONFIG_FILENAME
        self._runtime = get_runtime(work_path)
        self.lock = self._runtime.io_lock if self._runtime else threading.RLock()

    def read(self) -> PoolConfig:
        with self.lock:
            try:
                if not self.path.exists():
                    return PoolConfig()
                config = _parse(PoolConfig, json.loads(self.path.read_text(encoding="utf-8")))
                config.validate_references()
                return config
            except (OSError, ValueError, ConfigError):
                if self._runtime:
                    self._runtime.fail("invalid_llm_pool_storage")
                raise ConfigError("Cannot read LLM pool configuration; restore a valid backup", 503) from None

    def _write(self, config: PoolConfig):
        data = config.model_dump(mode="json")
        for item, model in zip(data["models"], config.models):
            item["api_key"] = model.api_key.get_secret_value()
        temp_name = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=".llm-pools-", dir=self.path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(data, output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            if self._runtime:
                from sirchmunk.storage.lifecycle import durable_replace
                durable_replace(Path(temp_name), self.path)
            else:
                os.replace(temp_name, self.path)
        except OSError:
            if self._runtime:
                self._runtime.fail("llm_pool_persistence_failed")
            raise ConfigError("Cannot save LLM pool configuration", 503) from None
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def change(self, revision: int, operation: str, payload=None, item_id=None) -> PoolConfig:
        with self.lock:
            config = self.read()
            if type(revision) is not int or config.revision != revision:
                raise ConfigError("Configuration changed; refresh before saving", 409)
            if operation in ("model", "pool"):
                if not isinstance(payload, dict):
                    raise ConfigError("Expected a model or pool object")
                collection = config.models if operation == "model" else config.pools
                old = next((m for m in collection if m.id == item_id), None)
                if item_id and old is None:
                    raise ConfigError("Model or pool not found", 404)
                values = old.model_dump() if old else {}
                update = dict(payload or {})
                if "id" in update:
                    raise ConfigError("ID is server assigned and cannot be edited")
                if operation == "model":
                    clear = update.pop("clear_api_key", False)
                    if type(clear) is not bool:
                        raise ConfigError("clear_api_key must be boolean")
                    key = update.get("api_key")
                    if key in (None, "", "***"):
                        update.pop("api_key", None)
                    if clear:
                        update.update(api_key="", api_key_env=None)
                    elif key and key != "***":
                        update.setdefault("api_key_env", None)
                values.update(update)
                values["id"] = item_id or uuid.uuid4().hex
                parsed = _parse(ModelSpec if operation == "model" else PoolSpec, values)
                if old:
                    collection[collection.index(old)] = parsed
                else:
                    if len(collection) >= (MAX_MODELS if operation == "model" else MAX_POOLS):
                        raise ConfigError("Configuration item limit reached")
                    collection.append(parsed)
            elif operation in ("delete_model", "delete_pool"):
                kind = operation.removeprefix("delete_")
                if config.active_target == Target(kind=kind, id=item_id):
                    raise ConfigError("Select another target before deleting the active item", 409)
                if kind == "model" and any(m.model_id == item_id for p in config.pools for m in p.members):
                    raise ConfigError("Remove model from its pools before deleting", 409)
                collection = config.models if kind == "model" else config.pools
                old = next((m for m in collection if m.id == item_id), None)
                if old is None:
                    raise ConfigError("Model or pool not found", 404)
                collection.remove(old)
            elif operation == "selection":
                config.active_target = _parse(Target, payload)
            else:
                raise ConfigError("Unknown configuration operation")
            config.validate_references()
            config.revision += 1
            self._write(config)
            return config


_stores: dict[str, ConfigStore] = {}
_stores_lock = threading.Lock()


def get_store() -> ConfigStore:
    path = str(Path(os.getenv("SIRCHMUNK_WORK_PATH", "~/.sirchmunk")).expanduser().resolve())
    with _stores_lock:
        if path not in _stores:
            _stores[path] = ConfigStore(path)
        return _stores[path]

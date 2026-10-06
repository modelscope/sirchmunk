"""LLM administration. Responses and validation errors are credential-free."""
import json
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from .security import verify_token
from sirchmunk.llm.pool_config import ConfigError, Target, _parse, get_store
from sirchmunk.llm.pool_router import PoolError, classify_error, get_scheduler
from sirchmunk.llm.factory import create_configured_llm, llm_request_scope, configuration_status

router = APIRouter(prefix="/api/v1/llm", tags=["llm-pools"], dependencies=[Depends(verify_token)])
MAX_BODY_BYTES = 128 * 1024


async def _body(request: Request) -> dict:
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > MAX_BODY_BYTES:
            raise HTTPException(413, "Configuration request too large")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError()
        return data
    except (ValueError, UnicodeError):
        raise HTTPException(422, "Expected a JSON object") from None


def _change(revision, operation, payload=None, item_id=None):
    try:
        store = get_store()
        config = store.change(revision, operation, payload, item_id)
        get_scheduler(str(store.path)).configure(config)
        return {"success": True, "data": config.public()}
    except ConfigError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@router.get("/config")
def get_config(response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return {"success": True, "data": get_store().read().public()}
    except ConfigError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@router.post("/models")
async def create_model(request: Request):
    body = await _body(request)
    return _change(body.get("revision"), "model", body.get("model"))


@router.post("/models/{item_id}")
async def update_model(item_id: str, request: Request):
    body = await _body(request)
    return _change(body.get("revision"), "model", body.get("model"), item_id)


@router.delete("/models/{item_id}")
async def delete_model(item_id: str, request: Request):
    body = await _body(request)
    return _change(body.get("revision"), "delete_model", item_id=item_id)


@router.post("/pools")
async def create_pool(request: Request):
    body = await _body(request)
    return _change(body.get("revision"), "pool", body.get("pool"))


@router.post("/pools/{item_id}")
async def update_pool(item_id: str, request: Request):
    body = await _body(request)
    return _change(body.get("revision"), "pool", body.get("pool"), item_id)


@router.delete("/pools/{item_id}")
async def delete_pool(item_id: str, request: Request):
    body = await _body(request)
    return _change(body.get("revision"), "delete_pool", item_id=item_id)


@router.post("/selection")
async def select_target(request: Request):
    body = await _body(request)
    return _change(body.get("revision"), "selection", body.get("target"))


@router.get("/status")
def get_status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        store = get_store()
        config = store.read()
        status = get_scheduler(str(store.path)).status(config)
        for model in status["models"]:
            model.pop("cooldown_until", None)
        return {"success": True, "data": {**status, "selection": configuration_status(config)}}
    except ConfigError as exc:
        raise HTTPException(exc.status, str(exc)) from None


async def test_configuration(config):
    """Tiny explicit probe. Never returns provider body, URL or credentials."""
    try:
        async with llm_request_scope(config):
            client = create_configured_llm()
            result = await asyncio.wait_for(client.achat(
                [{"role": "user", "content": "Reply with OK."}], stream=False, max_tokens=128,
            ), timeout=20)
        return {"success": True, "status": "configured", "message": "LLM connection successful",
                "model": result.model}
    except ConfigError as exc:
        return {"success": False, "status": "error", "message": str(exc), "model": None}
    except Exception as exc:
        code = exc.code if isinstance(exc, PoolError) else classify_error(exc)[0]
        return {"success": False, "status": "error", "message": code, "model": None}


@router.post("/test")
async def test_target(request: Request):
    body = await _body(request)
    try:
        config = get_store().read()
        if type(body.get("revision")) is not int or body["revision"] != config.revision:
            raise ConfigError("Configuration changed; refresh before testing", 409)
        if "target" in body:
            config.active_target = _parse(Target, body["target"])
        config.validate_references()
    except ConfigError as exc:
        raise HTTPException(exc.status, str(exc)) from None
    return await test_configuration(config)

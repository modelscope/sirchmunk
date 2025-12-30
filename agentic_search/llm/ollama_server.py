import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# Loguru Configuration (Async-safe)
logger.remove()
logger.add(
    "logs/ollama_proxy_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    enqueue=True,
)
logger.add(sys.stderr, level="INFO", colorize=True, enqueue=True)


# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "modelscope.cn/Qwen/Qwen3-0.6B-GGUF:latest"
DEFAULT_STREAM = True

# ⚙️ Tuning parameters (adjust based on your Ollama capacity)
MAX_OLLAMA_CONCURRENCY = 20  # Max concurrent requests to Ollama backend
HTTPX_MAX_CONNECTIONS = 100
HTTPX_MAX_KEEPALIVE = 50


# Reusable Async Client & Concurrency Control
OLLAMA_CLIENT = httpx.AsyncClient(
    base_url=OLLAMA_BASE_URL,
    timeout=httpx.Timeout(300.0, connect=10.0, read=290.0),
    limits=httpx.Limits(
        max_connections=HTTPX_MAX_CONNECTIONS,
        max_keepalive_connections=HTTPX_MAX_KEEPALIVE,
    ),
    # Enable HTTP/2 multiplexing
    http2=True,
)

OLLAMA_SEMAPHORE = asyncio.Semaphore(MAX_OLLAMA_CONCURRENCY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Ollama Proxy with optimized client & concurrency control")
    yield
    logger.info("Shutting down Ollama client...")
    await OLLAMA_CLIENT.aclose()


app = FastAPI(
    title="Ollama OpenAI-Compatible API Proxy (High-Throughput)",
    description="Proxy Ollama models to OpenAI API format with streaming, concurrency control, and metrics.",
    lifespan=lifespan,
)


# —————— Prometheus Metrics ——————
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    REQUESTS_TOTAL = Counter(
        "ollama_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
    )
    REQUEST_DURATION = Histogram(
        "ollama_request_duration_seconds", "HTTP request duration in seconds"
    )
    OLLAMA_CALLS_TOTAL = Counter(
        "ollama_backend_calls_total", "Total calls to Ollama backend", ["status"]
    )
except ImportError:
    logger.warning("prometheus_client not installed. /metrics endpoint disabled.")
    REQUESTS_TOTAL = REQUEST_DURATION = OLLAMA_CALLS_TOTAL = None


# —————— Middleware ——————
@app.middleware("http")
async def metrics_and_log_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = "5xx"

    # Log request
    logger.info(
        f"→ {request.method} {request.url.path} | client: {request.client.host}"
    )

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        raise
    finally:
        duration = (time.perf_counter() - start) * 1000  # ms
        logger.info(
            f"← {status} {request.method} {request.url.path} | {duration:.1f}ms"
        )
        if REQUESTS_TOTAL:
            REQUESTS_TOTAL.labels(request.method, request.url.path, status).inc()
        if REQUEST_DURATION:
            REQUEST_DURATION.observe(time.perf_counter() - start)


# —————— Pydantic Models ——————
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        default=DEFAULT_MODEL, description="The model to use for completion."
    )
    messages: List[Message] = Field(
        ..., description="A list of messages comprising the conversation."
    )
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None)
    stream: Optional[bool] = Field(default=DEFAULT_STREAM)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    seed: Optional[int] = Field(default=None)


# —————— Helper Functions ——————
async def call_ollama_non_stream(payload: dict) -> dict:
    try:
        async with OLLAMA_SEMAPHORE:
            resp = await OLLAMA_CLIENT.post("/api/chat", json=payload)
            if REQUESTS_TOTAL:
                OLLAMA_CALLS_TOTAL.labels(
                    "success" if resp.status_code == 200 else "error"
                ).inc()

            if resp.status_code != 200:
                error_detail = (
                    f"Ollama non-stream error {resp.status_code}: {resp.text[:200]}"
                )
                logger.error(error_detail)
                raise HTTPException(status_code=resp.status_code, detail=error_detail)
            return resp.json()
    except httpx.RequestError as e:
        logger.error(f"Ollama connection error (non-stream): {e}")
        if REQUESTS_TOTAL:
            OLLAMA_CALLS_TOTAL.labels("network_error").inc()
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {e}")


async def ollama_stream_generator(request: ChatCompletionRequest):
    payload = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stream": True,
        "options": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens or -1,
            "stop": request.stop or [],
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        },
    }
    if request.seed is not None:
        payload["options"]["seed"] = request.seed

    created_ts = int(time.time())
    # More robust chunk ID (avoid hash collision risk)
    chunk_id = f"chatcmpl-{created_ts}-{abs(hash(str(request.messages))) % 10000:04}"

    # First chunk: role=assistant
    first_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }

    # Safe yield: ignore client disconnect
    try:
        yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
    except Exception:
        return  # Client already gone

    try:
        async with OLLAMA_SEMAPHORE:
            async with OLLAMA_CLIENT.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    detail = f"Ollama stream error {resp.status_code}: {error_text.decode()[:200]}"
                    logger.error(detail)
                    if REQUESTS_TOTAL:
                        OLLAMA_CALLS_TOTAL.labels("stream_error").inc()
                    # Send error chunk + [DONE] via finally
                    raise HTTPException(status_code=resp.status_code, detail=detail)

                if REQUESTS_TOTAL:
                    OLLAMA_CALLS_TOTAL.labels("stream_start").inc()

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        ollama_chunk = json.loads(line)
                        content = ollama_chunk.get("message", {}).get("content", "")
                        done = ollama_chunk.get("done", False)

                        delta = {"content": content} if content else {}

                        openai_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "logprobs": None,
                                    "finish_reason": "stop" if done else None,
                                }
                            ],
                        }

                        # Safe yield inside loop
                        try:
                            yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                        except Exception:
                            # Client disconnected mid-stream
                            return

                        if done:
                            return  # Exit early — [DONE] handled in finally

                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Ignored invalid JSON line: {line[:100]} | error: {e}"
                        )
                        continue

    except httpx.RequestError as e:
        logger.error(f"Ollama stream connection error: {e}")
        if REQUESTS_TOTAL:
            OLLAMA_CALLS_TOTAL.labels("stream_conn_error").inc()
        # Do NOT raise — let finally send [DONE]
    except asyncio.CancelledError:
        logger.info("Client disconnected (stream cancelled)")
        # Clean exit: no raise, [DONE] in finally
    except Exception:
        logger.exception("Unexpected error in stream generator")
        # Let finally handle graceful termination
    finally:
        # CRITICAL: Always send final [DONE] to terminate SSE
        try:
            yield "data: [DONE]\n\n"
        except Exception:
            # Client likely disconnected — safe to ignore
            pass


# —————— API Endpoints ——————
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    logger.debug(
        f"Handling chat completion: model={request.model}, stream={request.stream}"
    )

    if request.stream:
        return StreamingResponse(
            ollama_stream_generator(request),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        payload = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens or -1,
                "stop": request.stop or [],
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
            },
        }
        if request.seed is not None:
            payload["options"]["seed"] = request.seed

        try:
            ollama_resp = await call_ollama_non_stream(payload)
            created = int(time.time())
            content = ollama_resp["message"]["content"]
            prompt_tokens = ollama_resp.get("prompt_eval_count", 0)
            completion_tokens = ollama_resp.get("eval_count", 0)

            openai_resp = {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            return JSONResponse(openai_resp)

        except Exception as e:
            logger.exception("Non-stream completion failed")
            raise HTTPException(status_code=500, detail=f"Backend error: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "max_concurrency": MAX_OLLAMA_CONCURRENCY,
        "uptime": (
            time.time() - app.start_time if hasattr(app, "start_time") else "unknown"
        ),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": int(time.time()) - 86400,
                "owned_by": "ollama",
            }
        ],
    }


# —————— Metrics Endpoint (if prometheus_client available) ——————
if REQUESTS_TOTAL is not None:

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

else:

    @app.get("/metrics")
    async def metrics_disabled():
        return JSONResponse(
            {"error": "prometheus_client not installed"}, status_code=501
        )


# —————— Startup Hook for Uptime ——————
@app.on_event("startup")
async def startup_event():
    app.start_time = time.time()
    logger.info("Ollama Proxy startup complete. Ready for high-throughput traffic.")

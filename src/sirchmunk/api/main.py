# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Main FastAPI application for Sirchmunk API
Combines all API modules and provides centralized configuration.

When the SIRCHMUNK_SERVE_UI environment variable is set to "true",
the application also serves the pre-built WebUI static files from
the Sirchmunk cache directory, enabling single-port access to both
the API and the WebUI.
"""

import os
from pathlib import Path

# Management identity and credentials must come from the process supervisor,
# never from the business-editable work-path .env file.
_management_launch_env = {
    key: os.environ.get(key)
    for key in (
        "SIRCHMUNK_WORK_PATH", "SIRCHMUNK_MANAGEMENT_ENABLED",
        "SIRCHMUNK_MANAGEMENT_CONFIG_ENABLED", "SIRCHMUNK_MANAGEMENT_INSTANCE_ID",
        "SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED",
        "SIRCHMUNK_RELEASE_ROOT", "SIRCHMUNK_MANAGEMENT_SHUTDOWN_TIMEOUT_SECONDS",
        "SIRCHMUNK_MANAGEMENT_PORT", "SIRCHMUNK_MANAGEMENT_TOKEN",
        "SIRCHMUNK_DEPLOYMENT_ID",
    )
}

def _protected_management_key(key: str) -> bool:
    upper = key.upper()
    return upper.startswith("SIRCHMUNK_MANAGEMENT_") or upper in {"SIRCHMUNK_DEPLOYMENT_ID", "SIRCHMUNK_RELEASE_ROOT"}

# Keep the independent management identity out of the business process env.
# In particular, dotenv interpolation of ordinary LLM/UI values must never
# expand the supervisor's management token.
for _key in tuple(os.environ):
    if _protected_management_key(_key):
        os.environ.pop(_key, None)

# Load .env file from Sirchmunk work directory before any module imports.
# This ensures environment variables (LLM_API_KEY, LLM_BASE_URL, etc.) are
# available when constants.py and other modules are first imported.
_work_path = Path(
    os.getenv("SIRCHMUNK_WORK_PATH", os.path.expanduser("~/.sirchmunk"))
).expanduser().resolve()
_env_file = _work_path / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(str(_env_file), override=False)
    except ImportError:
        # Fallback: manual .env parsing if python-dotenv is not installed
        try:
            with open(_env_file, "r") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _key, _, _val = _line.partition("=")
                        _key = _key.strip()
                        _val = _val.strip().strip('"').strip("'")
                        if _key and _key not in os.environ:
                            os.environ[_key] = _val
        except Exception:
            pass

for _key in tuple(os.environ):
    if _protected_management_key(_key):
        os.environ.pop(_key, None)

import logging

from sirchmunk.storage.lifecycle import bootstrap_managed_root

# Root ownership is acquired before any router constructs persistent storage.
_managed_runtime = bootstrap_managed_root(_management_launch_env)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse, Response

from .security import SecurityHeadersMiddleware, verify_token
from .management import AdmissionMiddleware, ManagementHolder, start_management, stop_management, close_managed_runtime

logger = logging.getLogger(__name__)

# Import all API routers
from .knowledge import router as knowledge_router
from .settings import router as settings_router
from .history import router as history_router, dashboard_router
from .chat import router as chat_router
from .monitor import router as monitor_router
from .search import router as search_router
from .files import router as files_router
from .llm_pools import router as llm_pools_router

# Determine whether to serve the WebUI static files.
# Set by `sirchmunk web serve` via environment variable.
_serve_ui = os.getenv("SIRCHMUNK_SERVE_UI") == "true"
_static_dir = _work_path / ".cache" / "web_static"
_ui_available = _serve_ui and _static_dir.is_dir() and (_static_dir / "index.html").exists()

# Create FastAPI application
_debug = os.getenv("SIRCHMUNK_DEBUG", "false").lower() == "true"

app = FastAPI(
    title="Sirchmunk API",
    description="APIs for Sirchmunk",
    version="1.0.0",
    docs_url="/docs" if _debug else None,
    redoc_url="/redoc" if _debug else None,
)

_management_holder = ManagementHolder()
app.state.management_holder = _management_holder
app.add_middleware(AdmissionMiddleware, holder=_management_holder)

# Configure CORS
_allowed_origins_raw = os.getenv("SIRCHMUNK_ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
if not _allowed_origins:
    _allowed_origins = ["*"]  # backward-compatible when unconfigured

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.add_middleware(SecurityHeadersMiddleware)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Enforce Bearer-token auth on /api/ routes when SIRCHMUNK_API_TOKEN is set."""
    path = request.url.path
    # Exempt: health check, favicon, static assets, OPTIONS preflight, UI settings
    exempt = (
        path in ("/health", "/favicon.ico")
        or path.startswith("/_next/")
        or (path == "/api/v1/settings/ui" and request.method == "GET")
        or request.method == "OPTIONS"
    )
    if not exempt and path.startswith("/api/"):
        try:
            await verify_token(request)
        except HTTPException as exc:
            # Exceptions raised from user middleware sit outside FastAPI's
            # route exception handler; return the intended HTTP status here.
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    response = await call_next(request)
    return response


@app.on_event("startup")
async def _start_management_listener():
    await start_management(_management_holder, _work_path, _management_launch_env)
    if _managed_runtime:
        import asyncio
        try:
            await asyncio.to_thread(_management_holder.state.require_lifecycle().initialize)
        except BaseException:
            await stop_management(_management_holder)
            raise


@app.on_event("startup")
async def _prewarm_chat_search():
    """Warm cached embedding resources in a short-lived request scope at startup.
    This reduces the chance of the first user request blocking on model load (e.g. in Docker).
    """
    if _managed_runtime:
        # Managed readiness proves local storage, not remote model availability.
        # Embeddings load offline on demand; startup must not download models.
        return
    try:
        from .chat import get_search_instance
        from sirchmunk.llm.factory import llm_request_scope
        async with llm_request_scope():
            get_search_instance()
    except Exception:
        pass


@app.on_event("shutdown")
async def _stop_management_listener():
    try:
        if _managed_runtime:
            timeout = float(_management_launch_env.get("SIRCHMUNK_MANAGEMENT_SHUTDOWN_TIMEOUT_SECONDS") or 60)
            if not 1 <= timeout <= 600:
                raise RuntimeError("invalid_managed_shutdown_timeout")
            await close_managed_runtime(_management_holder, timeout)
    finally:
        await stop_management(_management_holder)


# Include all API routers (registered before static mount so they take priority)
app.include_router(knowledge_router)
app.include_router(settings_router)
app.include_router(history_router)
app.include_router(dashboard_router)
app.include_router(chat_router)
app.include_router(monitor_router)
app.include_router(search_router)
app.include_router(files_router)
app.include_router(llm_pools_router)

# Root endpoint: return API info when UI is not served,
# otherwise let the static mount handle "/"
if not _ui_available:
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "Sirchmunk API",
            "version": "1.0.0",
            "description": "APIs for Sirchmunk",
            "status": "running",
            "endpoints": {
                "search": "/api/v1/search",
                "knowledge": "/api/v1/knowledge",
                "settings": "/api/v1/settings",
                "history": "/api/v1/history",
                "chat": "/api/v1/chat",
                "monitor": "/api/v1/monitor"
            },
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ui_enabled": _ui_available,
        "services": {
            "api": "running",
            "database": "connected",
            "llm": "available",
            "embedding": "available"
        }
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred",
                "details": "Please try again later or contact support"
            }
        }
    )

# Favicon route — browsers always request /favicon.ico; serve it from the
# static directory when available, otherwise return 204 No Content to avoid 404.
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if _ui_available:
        ico_path = _static_dir / "favicon.ico"
        if ico_path.is_file():
            return FileResponse(ico_path, media_type="image/x-icon")
    return Response(status_code=204)


# Mount static files for WebUI when enabled.
# This MUST be after all API route registrations so that API endpoints
# take priority over the catch-all static file serving.
if _ui_available:
    from fastapi.staticfiles import StaticFiles

    # SPA route fallback — Next.js static export creates route directories
    # (e.g. history/) for RSC payloads but no index.html inside them.
    # Starlette's StaticFiles(html=True) resolves the directory first and
    # returns 404 when index.html is missing. This middleware intercepts
    # those 404s for known frontend routes and serves the correct .html file.
    _SPA_ROUTES = {"history", "knowledge", "monitor", "settings"}

    @app.middleware("http")
    async def spa_fallback(request: Request, call_next):
        response = await call_next(request)
        if response.status_code == 404:
            path = request.url.path.strip("/").split("/")[0]
            if path in _SPA_ROUTES:
                html_file = _static_dir / f"{path}.html"
                if html_file.is_file():
                    if request.method == "HEAD":
                        return Response(
                            status_code=200,
                            headers={"content-type": "text/html; charset=utf-8"},
                        )
                    return FileResponse(html_file, media_type="text/html")
        return response

    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="ui")
    logger.info("WebUI enabled, serving static files from %s", _static_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8584,
        reload=True,
        log_level="info"
    )

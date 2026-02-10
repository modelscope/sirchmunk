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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import all API routers
from .knowledge import router as knowledge_router
from .settings import router as settings_router
from .history import router as history_router, dashboard_router
from .chat import router as chat_router
from .monitor import router as monitor_router
from .search import router as search_router

# Determine whether to serve the WebUI static files.
# Set by `sirchmunk web serve` via environment variable.
_serve_ui = os.getenv("SIRCHMUNK_SERVE_UI") == "true"
_static_dir = _work_path / ".cache" / "web_static"
_ui_available = _serve_ui and _static_dir.is_dir() and (_static_dir / "index.html").exists()

# Create FastAPI application
app = FastAPI(
    title="Sirchmunk API",
    description="APIs for Sirchmunk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routers (registered before static mount so they take priority)
app.include_router(knowledge_router)
app.include_router(settings_router)
app.include_router(history_router)
app.include_router(dashboard_router)
app.include_router(chat_router)
app.include_router(monitor_router)
app.include_router(search_router)

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

# Mount static files for WebUI when enabled.
# This MUST be after all API route registrations so that API endpoints
# take priority over the catch-all static file serving.
if _ui_available:
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="ui")
    print(f"[INFO] WebUI enabled, serving static files from {_static_dir}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8584,
        reload=True,
        log_level="info"
    )

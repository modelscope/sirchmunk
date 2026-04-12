"""Security utilities for Sirchmunk API: authentication, path validation,
prompt-injection detection, filename sanitization, and HTTP security headers."""

import hmac
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException, Request, WebSocket, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token Authentication
# ---------------------------------------------------------------------------

def _get_api_token() -> Optional[str]:
    """Read and normalize API token from environment on each call."""
    raw = os.getenv("SIRCHMUNK_API_TOKEN")
    if raw is None:
        return None
    token = raw.strip()
    return token or None


async def verify_token(request: Request) -> None:
    """Verify Bearer token. No-op when SIRCHMUNK_API_TOKEN is unset."""
    token = _get_api_token()
    if not token:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )
    presented = auth[7:].strip()
    if not hmac.compare_digest(presented, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )


def verify_ws_token(websocket: WebSocket) -> bool:
    """Verify WebSocket token from query param or Authorization header."""
    token = _get_api_token()
    if not token:
        return True
    candidate = websocket.query_params.get("token", "")
    if not candidate:
        auth = websocket.headers.get("authorization", "")
        candidate = auth[7:].strip() if auth.startswith("Bearer ") else ""
    return bool(candidate) and hmac.compare_digest(candidate, token)


# ---------------------------------------------------------------------------
# Path Whitelist
# ---------------------------------------------------------------------------


def get_allowed_paths() -> List[Path]:
    """Return resolved allowed paths from env + uploads directory."""
    raw = os.getenv("SIRCHMUNK_ALLOWED_PATHS", "")
    work_path = os.getenv("SIRCHMUNK_WORK_PATH", os.path.expanduser("~/.sirchmunk"))
    paths = [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]
    # Always allow the uploads directory
    paths.append(Path(work_path).resolve() / "uploads")
    return paths


def is_path_allowed(requested: str) -> bool:
    """Check whether *requested* falls under an allowed base path.

    When SIRCHMUNK_ALLOWED_PATHS is unset, all paths are allowed (backward-compat).
    """
    env_raw = os.getenv("SIRCHMUNK_ALLOWED_PATHS", "")
    if not env_raw.strip():
        return True  # unrestricted when unconfigured
    allowed = get_allowed_paths()
    target = Path(requested).resolve()
    return any(_is_subpath(target, base) for base in allowed)


def _is_subpath(child: Path, parent: Path) -> bool:
    """Return True if *child* is equal to or a descendant of *parent*."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

# ---------------------------------------------------------------------------
# Filename Sanitization
# ---------------------------------------------------------------------------

def sanitize_filename(filename: str) -> str:
    """Strip path components and dangerous characters from *filename*."""
    name = os.path.basename(filename)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    if not name or name.startswith('.'):
        name = f"unnamed_{name}"
    return name


# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject standard security headers into every HTTP response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' ws: wss:; "
            "font-src 'self' data:;",
        )
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=63072000; includeSubDomains",
        )
        response.headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=()",
        )
        return response

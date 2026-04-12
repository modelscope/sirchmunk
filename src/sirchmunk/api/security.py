"""Security utilities for Sirchmunk API: authentication, path validation,
prompt-injection detection, filename sanitization, and HTTP security headers."""

import hmac
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
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
    work_path_resolved = Path(work_path).resolve()
    paths = [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]
    # Always allow the uploads directory
    uploads_path = work_path_resolved / "uploads"
    if uploads_path not in paths:
        paths.append(uploads_path)
    return paths


def _has_symlink_in_chain(path: str) -> bool:
    """Check whether any component of *path* is a symbolic link."""
    try:
        p = Path(path)
        for component in [p] + list(p.parents):
            if component.is_symlink():
                return True
        return False
    except (OSError, ValueError):
        return True  # Fail-closed on errors


def is_path_allowed(requested: str) -> bool:
    """Check whether *requested* falls under an allowed base path.

    When SIRCHMUNK_ALLOWED_PATHS is unset, all paths are allowed (backward-compat).
    """
    env_raw = os.getenv("SIRCHMUNK_ALLOWED_PATHS", "")
    if not env_raw.strip():
        return True  # unrestricted when unconfigured
    # Reject paths containing symbolic links
    if _has_symlink_in_chain(requested):
        logger.warning("Symlink detected in path: %s", requested)
        return False
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


def validate_user_path(user_path: str) -> tuple:
    """Validate a user-provided search path.

    Returns (is_valid: bool, result: str) where *result* is either the
    canonicalised path on success or a generic error description on failure.
    """
    if not user_path or not user_path.strip():
        return False, "Path cannot be empty"

    user_path = user_path.strip()

    # Reject relative path components
    if ".." in user_path or user_path.startswith("~"):
        return False, "Relative paths not allowed"

    # Reject shell-dangerous characters
    if any(c in user_path for c in "`;&|$(){}"):
        return False, "Invalid characters in path"

    # Normalise
    try:
        abs_path = os.path.abspath(user_path)
        real_path = os.path.realpath(abs_path)
    except (ValueError, OSError):
        return False, "Invalid path format"

    # Symlink detection
    if abs_path != real_path:
        logger.warning("Symlink detected in user path: %s -> %s", abs_path, real_path)
        return False, "Access denied"

    # Whitelist check
    if not is_path_allowed(real_path):
        return False, "Access denied"

    # Existence + directory check
    if not os.path.exists(real_path):
        return False, "Path does not exist"
    if not os.path.isdir(real_path):
        return False, "Path must be a directory"

    return True, real_path


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


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple in-memory per-IP rate limiter (token-bucket style)."""

    def __init__(self, per_second: int = 5, per_minute: int = 100):
        self._per_second = per_second
        self._per_minute = per_minute
        self._hits: dict = defaultdict(list)  # ip -> [timestamps]

    def is_allowed(self, client_ip: str) -> bool:
        """Return True if the request from *client_ip* is within limits."""
        now = time.time()
        history = self._hits[client_ip]

        # Prune entries older than 60 seconds
        cutoff = now - 60
        self._hits[client_ip] = [t for t in history if t > cutoff]
        history = self._hits[client_ip]

        # Per-second check
        recent = sum(1 for t in history if now - t < 1)
        if recent >= self._per_second:
            return False

        # Per-minute check
        if len(history) >= self._per_minute:
            return False

        history.append(now)
        return True


# Shared rate-limiter instance for file-browser endpoint
file_browser_limiter = RateLimiter(per_second=5, per_minute=100)


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------


class AuditLogger:
    """Append-only JSON-Lines audit log for path access events."""

    def __init__(self):
        work_path = os.getenv("SIRCHMUNK_WORK_PATH", os.path.expanduser("~/.sirchmunk"))
        log_dir = Path(work_path).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = log_dir / "audit.log"

    def log(self, *, client_ip: str, action: str, path: str, result: str) -> None:
        """Write a single audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "client_ip": client_ip,
            "action": action,
            "path": path,
            "result": result,
        }
        try:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Failed to write audit log entry")


audit_logger = AuditLogger()

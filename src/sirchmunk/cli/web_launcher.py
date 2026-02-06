# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Web UI launcher for Sirchmunk.

Manages frontend build (Next.js static export) and dev server lifecycle.
Supports two modes:
- Production: Pre-built static files served by FastAPI (single port)
- Development: Next.js dev server as subprocess (dual port, hot reload)
"""

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Default static build output directory inside Sirchmunk work path
_STATIC_CACHE_DIR = ".cache/web_static"

# Environment variable name used by Next.js to embed the API base URL
_NEXT_API_BASE_ENV = "NEXT_PUBLIC_API_BASE"

# Build mode environment variable for next.config.js
_NEXT_BUILD_STATIC_ENV = "NEXT_BUILD_STATIC"


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def _print(msg: str):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def check_node_installed() -> bool:
    """Check if Node.js and npm are available on the system.

    Returns:
        True if both node and npm are found in PATH
    """
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        return False
    return True


def get_node_version() -> Optional[str]:
    """Get the installed Node.js version string, or None."""
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def find_web_source_dir() -> Optional[Path]:
    """Locate the Next.js web source directory.

    Search order:
    1. ``<project_root>/web`` (development / git clone)
    2. ``<package_root>/../../../web`` (pip-installed package)

    Returns:
        Path to ``web/`` directory, or None if not found
    """
    # Strategy 1: relative to this file -> project root / web
    pkg_root = Path(__file__).resolve().parent.parent  # src/sirchmunk
    project_root = pkg_root.parent.parent  # workspace root
    candidate = project_root / "web"
    if candidate.is_dir() and (candidate / "package.json").exists():
        return candidate

    # Strategy 2: environment variable override
    env_web = os.getenv("SIRCHMUNK_WEB_DIR")
    if env_web:
        p = Path(env_web).expanduser().resolve()
        if p.is_dir() and (p / "package.json").exists():
            return p

    return None


def get_static_dir(work_path: Path) -> Path:
    """Get the path to the cached static build output.

    Args:
        work_path: Sirchmunk work directory (e.g. ``~/.sirchmunk``)

    Returns:
        Path to static files directory
    """
    return work_path / _STATIC_CACHE_DIR


def has_static_build(work_path: Path) -> bool:
    """Check if a valid static build exists in the cache.

    Args:
        work_path: Sirchmunk work directory

    Returns:
        True if the cached static build contains an index.html
    """
    static_dir = get_static_dir(work_path)
    return (static_dir / "index.html").exists()


# ---------------------------------------------------------------------------
#  Frontend build (static export)
# ---------------------------------------------------------------------------

def build_frontend(
    work_path: Path,
    web_dir: Optional[Path] = None,
    backend_port: int = 8584,
) -> bool:
    """Build the Next.js frontend as a static export.

    Runs ``npm install`` followed by ``npm run build`` with the
    ``NEXT_BUILD_STATIC=true`` environment variable to trigger
    ``output: 'export'`` in next.config.js. The build output is then
    copied to the Sirchmunk cache directory.

    Args:
        work_path: Sirchmunk work directory (e.g. ``~/.sirchmunk``)
        web_dir: Path to the ``web/`` source directory. Auto-detected if None.
        backend_port: Backend port for API base URL configuration

    Returns:
        True on success, False on failure
    """
    if web_dir is None:
        web_dir = find_web_source_dir()
    if web_dir is None:
        _print("  ✗ Cannot locate web/ source directory.")
        _print("    Set SIRCHMUNK_WEB_DIR environment variable if needed.")
        return False

    _print(f"  Web source: {web_dir}")

    # Check Node.js availability
    if not check_node_installed():
        _print("  ✗ Node.js or npm is not installed.")
        _print("    Install from: https://nodejs.org/")
        return False

    node_ver = get_node_version()
    _print(f"  ✓ Node.js {node_ver} detected")

    # Step 1: npm install
    _print("  Installing frontend dependencies...")
    npm_cmd = shutil.which("npm") or "npm"
    try:
        result = subprocess.run(
            [npm_cmd, "install"],
            cwd=str(web_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            # Check if node_modules was partially created
            if not (web_dir / "node_modules").exists():
                _print(f"  ✗ npm install failed: {result.stderr[:500]}")
                return False
            _print("  ⚠ npm install completed with warnings")
        else:
            _print("  ✓ Dependencies installed")
    except subprocess.TimeoutExpired:
        _print("  ✗ npm install timed out (300s)")
        return False
    except Exception as e:
        _print(f"  ✗ npm install failed: {e}")
        return False

    # Step 2: npm run build (with static export flag)
    _print("  Building static frontend (this may take a minute)...")
    build_env = os.environ.copy()
    build_env[_NEXT_BUILD_STATIC_ENV] = "true"
    # Empty API base = same-origin requests when served by FastAPI
    build_env[_NEXT_API_BASE_ENV] = ""

    try:
        result = subprocess.run(
            [npm_cmd, "run", "build"],
            cwd=str(web_dir),
            capture_output=True,
            text=True,
            env=build_env,
            timeout=600,
        )
        if result.returncode != 0:
            _print(f"  ✗ Build failed:")
            # Print last 20 lines of stderr/stdout for diagnostics
            output = (result.stderr or result.stdout or "").strip()
            for line in output.split("\n")[-20:]:
                _print(f"    {line}")
            return False
        _print("  ✓ Static build completed")
    except subprocess.TimeoutExpired:
        _print("  ✗ Build timed out (600s)")
        return False
    except Exception as e:
        _print(f"  ✗ Build failed: {e}")
        return False

    # Step 3: Copy build output to cache directory
    build_output = web_dir / "out"
    if not build_output.is_dir():
        _print("  ✗ Build output directory not found (web/out/)")
        _print("    Ensure next.config.js supports output: 'export'")
        return False

    static_dir = get_static_dir(work_path)
    _print(f"  Copying build output to {static_dir}...")
    try:
        if static_dir.exists():
            shutil.rmtree(static_dir)
        shutil.copytree(build_output, static_dir)

        # Fix SPA routing: Next.js static export creates directories (e.g. monitor/)
        # alongside HTML files (e.g. monitor.html) for RSC payloads. Starlette's
        # StaticFiles(html=True) resolves the directory first and looks for
        # {dir}/index.html, which doesn't exist, causing 404 errors. We fix
        # this by copying each {route}.html into {route}/index.html so that
        # both GET and HEAD requests resolve correctly.
        _fix_static_route_dirs(static_dir)

        _print(f"  ✓ Static files cached at {static_dir}")
    except Exception as e:
        _print(f"  ✗ Failed to copy build output: {e}")
        return False

    return True


def _fix_static_route_dirs(static_dir: Path):
    """Fix SPA route directories for StaticFiles compatibility.

    For each ``{route}.html`` file at the top level, if a corresponding
    ``{route}/`` directory exists without an ``index.html``, copy the HTML
    file into the directory as ``index.html``. This ensures Starlette's
    ``StaticFiles(html=True)`` correctly serves pages for paths like
    ``/monitor``, ``/history``, ``/knowledge``, ``/settings``, etc.

    Args:
        static_dir: Root directory of the static build output
    """
    for html_file in static_dir.glob("*.html"):
        route_name = html_file.stem  # e.g. "monitor" from "monitor.html"
        route_dir = static_dir / route_name
        if route_dir.is_dir() and not (route_dir / "index.html").exists():
            shutil.copy2(html_file, route_dir / "index.html")
            logger.debug(f"Created {route_dir / 'index.html'} for SPA routing")


# ---------------------------------------------------------------------------
#  Process management (development mode)
# ---------------------------------------------------------------------------

def terminate_process_tree(process, name: str = "Process", timeout: int = 5):
    """Terminate a subprocess and all its children.

    On Unix, uses process group (PGID) to kill child processes.
    On Windows, uses taskkill /T for the process tree.

    Args:
        process: subprocess.Popen object
        name: Display name for logging
        timeout: Seconds to wait before SIGKILL
    """
    if process is None or process.poll() is not None:
        return

    pid = process.pid
    _print(f"  Stopping {name} (PID: {pid})...")

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False, capture_output=True, text=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        else:
            pgid = os.getpgid(pid)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return

            try:
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                pass

            try:
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=2)
            except (ProcessLookupError, Exception):
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass
    except Exception as e:
        _print(f"  ⚠ Error stopping {name}: {e}")


def start_frontend_dev(
    web_dir: Path,
    frontend_port: int = 8585,
    backend_port: int = 8584,
) -> Optional[subprocess.Popen]:
    """Start the Next.js development server as a subprocess.

    Args:
        web_dir: Path to the ``web/`` source directory
        frontend_port: Port for the Next.js dev server
        backend_port: Backend port for API proxy configuration

    Returns:
        subprocess.Popen object, or None on failure
    """
    npm_cmd = shutil.which("npm") or "npm"

    env = os.environ.copy()
    env["PORT"] = str(frontend_port)
    env[_NEXT_API_BASE_ENV] = f"http://localhost:{backend_port}"
    env["PYTHONIOENCODING"] = "utf-8"

    popen_kwargs = {
        "cwd": str(web_dir),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "shell": False,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }

    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        process = subprocess.Popen(
            [npm_cmd, "run", "dev", "--", "-p", str(frontend_port)],
            **popen_kwargs,
        )
    except Exception as e:
        _print(f"  ✗ Failed to start frontend dev server: {e}")
        return None

    # Log output in background thread
    def _log_output():
        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    _print(f"[Frontend] {line.rstrip()}")
        except Exception:
            pass

    t = threading.Thread(target=_log_output, daemon=True)
    t.start()

    _print(f"  ✓ Frontend dev server started (PID: {process.pid})")
    return process


def start_backend_subprocess(
    backend_port: int = 8584,
    host: str = "0.0.0.0",
    log_level: str = "info",
    reload: bool = False,
) -> Optional[subprocess.Popen]:
    """Start the FastAPI backend as a subprocess.

    Used in development mode (--dev) where both frontend and backend
    run as child processes managed by the CLI.

    Args:
        backend_port: Port for the uvicorn server
        host: Host to bind
        log_level: Uvicorn log level
        reload: Enable auto-reload

    Returns:
        subprocess.Popen object, or None on failure
    """
    cmd = [
        sys.executable, "-m", "uvicorn",
        "sirchmunk.api.main:app",
        "--host", host,
        "--port", str(backend_port),
        "--log-level", log_level,
    ]
    if reload:
        cmd.append("--reload")

    # Resolve src directory for PYTHONPATH
    pkg_root = Path(__file__).resolve().parent.parent  # src/sirchmunk
    src_dir = str(pkg_root.parent)  # src/

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "shell": False,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }

    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        process = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        _print(f"  ✗ Failed to start backend: {e}")
        return None

    # Log output in background thread
    def _log_output():
        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    _print(f"[Backend]  {line.rstrip()}")
        except Exception:
            pass

    t = threading.Thread(target=_log_output, daemon=True)
    t.start()

    _print(f"  ✓ Backend started (PID: {process.pid})")
    return process


def wait_for_port(port: int, host: str = "localhost", timeout: int = 15) -> bool:
    """Wait until a port becomes available.

    Args:
        port: Port number to check
        host: Host to check
        timeout: Maximum seconds to wait

    Returns:
        True if port is reachable within timeout
    """
    import socket

    for _ in range(timeout * 2):
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
#  High-level orchestrators
# ---------------------------------------------------------------------------

def serve_with_ui_dev(
    host: str = "0.0.0.0",
    backend_port: int = 8584,
    frontend_port: int = 8585,
    log_level: str = "info",
):
    """Start both backend and frontend in development mode.

    Backend runs as a uvicorn subprocess; frontend runs as a Next.js
    dev server subprocess. The main process monitors both and handles
    graceful shutdown on Ctrl+C.

    Args:
        host: Host to bind the backend
        backend_port: Backend API port
        frontend_port: Frontend dev server port
        log_level: Logging level
    """
    web_dir = find_web_source_dir()
    if web_dir is None:
        _print("✗ Cannot locate web/ source directory for dev mode.")
        _print("  Set SIRCHMUNK_WEB_DIR environment variable.")
        return

    if not check_node_installed():
        _print("✗ Node.js / npm is required for --dev mode.")
        _print("  Install from: https://nodejs.org/")
        return

    backend = None
    frontend = None

    try:
        _print("Starting backend...")
        backend = start_backend_subprocess(
            backend_port=backend_port,
            host=host,
            log_level=log_level,
            reload=True,
        )
        if backend is None:
            return

        _print(f"Waiting for backend on port {backend_port}...")
        if not wait_for_port(backend_port, timeout=20):
            _print("✗ Backend did not start in time.")
            return

        _print(f"✓ Backend ready on http://localhost:{backend_port}")
        _print()

        _print("Starting frontend dev server...")
        frontend = start_frontend_dev(
            web_dir=web_dir,
            frontend_port=frontend_port,
            backend_port=backend_port,
        )
        if frontend is None:
            return

        _print()
        _print("=" * 60)
        _print("✅ Development servers running!")
        _print("=" * 60)
        _print(f"  Frontend: http://localhost:{frontend_port}")
        _print(f"  Backend:  http://localhost:{backend_port}")
        _print(f"  API Docs: http://localhost:{backend_port}/docs")
        _print("=" * 60)
        _print()
        _print("Press Ctrl+C to stop all services.")

        # Monitor loop
        while True:
            if backend.poll() is not None:
                _print(f"\n✗ Backend exited (code: {backend.returncode})")
                break
            if frontend.poll() is not None:
                _print(f"\n✗ Frontend exited (code: {frontend.returncode})")
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        _print("\nStopping services...")
    finally:
        terminate_process_tree(frontend, name="Frontend")
        terminate_process_tree(backend, name="Backend")
        _print("✅ All services stopped.")

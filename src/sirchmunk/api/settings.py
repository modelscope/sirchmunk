# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Settings API endpoints with .env-based persistent storage.
Provides UI settings and environment variable management.

All configuration is read from and written to the .env file
(located at {SIRCHMUNK_WORK_PATH}/.env).  At startup the .env
file is loaded into os.environ by main.py, so every read goes
through os.getenv() and every write goes through
_update_env_file() + os.environ.
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

# Default values
_DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LLM_MODEL_NAME = "gpt-5.2"
_DEFAULT_GREP_CONCURRENT_LIMIT = "5"
_DEFAULT_WORK_PATH = os.path.expanduser("~/.sirchmunk")

# Keys that must have non-empty values in target .env to "load and reuse" when switching work path
_REQUIRED_ENV_KEYS_FOR_REUSE = ("LLM_API_KEY", "LLM_BASE_URL")


def _get_env_file_path() -> Path:
    """Get the .env file path in the Sirchmunk work directory."""
    work_path = os.getenv("SIRCHMUNK_WORK_PATH", _DEFAULT_WORK_PATH)
    return Path(work_path).expanduser().resolve() / ".env"


def _load_env_file_to_dict(env_path: Path) -> Dict[str, str]:
    """Load a .env file into a key-value dict without modifying os.environ.

    Used when switching work path so we can preserve the existing .env
    at the target path instead of overwriting with defaults.

    Args:
        env_path: Path to the .env file.

    Returns:
        Dict of key -> value. Empty dict if file does not exist or on parse error.
    """
    out: Dict[str, str] = {}
    if not env_path.exists():
        return out
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key:
                if len(val) >= 2 and (val[0] == val[-1] == '"' or val[0] == val[-1] == "'"):
                    val = val[1:-1]
                out[key] = val
    except Exception:
        pass
    return out


def _existing_env_can_reuse(existing: Dict[str, str]) -> bool:
    """Return True if existing .env has all required keys with non-empty values."""
    for key in _REQUIRED_ENV_KEYS_FOR_REUSE:
        if not (existing.get(key) or "").strip():
            return False
    return True


def _update_env_file(updates: Dict[str, str]):
    """Update specific key-value pairs in the .env file.

    Preserves comments, blank lines, and overall file structure.
    Only updates existing keys or appends new ones at the end.
    Creates the file if it does not exist.

    Args:
        updates: Dictionary of key-value pairs to update
    """
    env_path = _get_env_file_path()

    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)

        if not env_path.exists():
            lines_out = [f"{k}={v}" for k, v in updates.items()]
            env_path.write_text("\n".join(lines_out) + "\n")
            return

        lines = env_path.read_text().splitlines()
        updated_keys: set = set()
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue

            if "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        for key, value in updates.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}")

        env_path.write_text("\n".join(new_lines) + "\n")
    except Exception as e:
        print(f"[WARNING] Failed to update .env file: {e}")

# === Request/Response Models ===

class UISettings(BaseModel):
    theme: str = "light"
    language: str = "en"

class EnvironmentVariables(BaseModel):
    SIRCHMUNK_WORK_PATH: Optional[str] = None
    SIRCHMUNK_SEARCH_PATHS: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: Optional[str] = None
    GREP_CONCURRENT_LIMIT: Optional[int] = None

class SaveSettingsRequest(BaseModel):
    ui: Optional[UISettings] = None
    environment: Optional[Dict[str, str]] = None

# === Helper Functions ===

def get_default_ui_settings() -> Dict[str, Any]:
    """Get UI settings from os.environ (backed by .env)."""
    return {
        "theme": os.getenv("UI_THEME", "light"),
        "language": os.getenv("UI_LANGUAGE", "en"),
    }

def get_current_env_variables() -> Dict[str, Any]:
    """Get current environment variables from os.environ (backed by .env)."""
    return {
        "SIRCHMUNK_WORK_PATH": {
            "value": os.getenv("SIRCHMUNK_WORK_PATH", _DEFAULT_WORK_PATH),
            "default": _DEFAULT_WORK_PATH,
            "description": "Working directory for Sirchmunk data",
            "category": "system"
        },
        "SIRCHMUNK_SEARCH_PATHS": {
            "value": os.getenv("SIRCHMUNK_SEARCH_PATHS", ""),
            "default": "",
            "description": "Default search paths (comma-separated). "
                           "Overridden by explicit paths passed to search().",
            "category": "search"
        },
        "LLM_BASE_URL": {
            "value": os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL),
            "default": _DEFAULT_LLM_BASE_URL,
            "description": "Base URL for LLM API (OpenAI-compatible endpoint)",
            "category": "llm"
        },
        "LLM_API_KEY": {
            "value": os.getenv("LLM_API_KEY", ""),
            "default": "",
            "description": "API key for LLM service",
            "category": "llm",
            "sensitive": True
        },
        "LLM_MODEL_NAME": {
            "value": os.getenv("LLM_MODEL_NAME", _DEFAULT_LLM_MODEL_NAME),
            "default": _DEFAULT_LLM_MODEL_NAME,
            "description": "Model name for LLM",
            "category": "llm"
        },
        "GREP_CONCURRENT_LIMIT": {
            "value": os.getenv("GREP_CONCURRENT_LIMIT", _DEFAULT_GREP_CONCURRENT_LIMIT),
            "default": _DEFAULT_GREP_CONCURRENT_LIMIT,
            "description": "Maximum concurrent grep requests",
            "category": "system"
        }
    }

# === API Endpoints ===

@router.get("")
async def get_all_settings():
    """Get all settings including UI and environment variables"""
    try:
        ui_settings = get_default_ui_settings()
        env_variables = get_current_env_variables()

        return {
            "success": True,
            "data": {
                "ui": ui_settings,
                "environment": env_variables
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ui")
async def get_ui_settings():
    """Get UI settings"""
    try:
        ui_settings = get_default_ui_settings()
        return {
            "success": True,
            "data": ui_settings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/environment")
async def get_environment_variables():
    """Get environment variables"""
    try:
        env_variables = get_current_env_variables()
        return {
            "success": True,
            "data": env_variables
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("")
async def save_settings(request: SaveSettingsRequest):
    """Save settings (UI and/or environment variables)"""
    try:
        saved_items = []
        env_updates: Dict[str, str] = {}

        # Save UI settings
        if request.ui:
            if request.ui.theme:
                env_updates["UI_THEME"] = request.ui.theme
                saved_items.append("theme")
            if request.ui.language:
                env_updates["UI_LANGUAGE"] = request.ui.language
                saved_items.append("language")

        # Save environment variables
        if request.environment:
            for key, value in request.environment.items():
                if value and value != "***":
                    env_updates[key] = str(value)
                    saved_items.append(key)

        if env_updates:
            # Only trigger work-path-switch logic when the value actually changed
            work_path_changed = False
            if "SIRCHMUNK_WORK_PATH" in env_updates:
                new_resolved = str(Path(env_updates["SIRCHMUNK_WORK_PATH"]).expanduser().resolve())
                old_resolved = str(Path(os.getenv("SIRCHMUNK_WORK_PATH", _DEFAULT_WORK_PATH)).expanduser().resolve())
                work_path_changed = (new_resolved != old_resolved)
                env_updates["SIRCHMUNK_WORK_PATH"] = new_resolved

            if work_path_changed:
                os.environ["SIRCHMUNK_WORK_PATH"] = env_updates["SIRCHMUNK_WORK_PATH"]
                new_env_path = Path(env_updates["SIRCHMUNK_WORK_PATH"]) / ".env"
                existing_at_new = _load_env_file_to_dict(new_env_path)

                if new_env_path.exists() and _existing_env_can_reuse(existing_at_new):
                    merged = dict(existing_at_new)
                    merged["SIRCHMUNK_WORK_PATH"] = env_updates["SIRCHMUNK_WORK_PATH"]
                    _update_env_file(merged)
                    for k, v in merged.items():
                        os.environ[k] = v
                else:
                    _update_env_file(env_updates)
                    for k, v in env_updates.items():
                        os.environ[k] = v
            else:
                _update_env_file(env_updates)
                for k, v in env_updates.items():
                    os.environ[k] = v

        return {
            "success": True,
            "message": f"Settings saved successfully: {', '.join(saved_items)}",
            "saved_items": saved_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")

@router.post("/ui")
async def update_ui_settings(ui: UISettings):
    """Update UI settings"""
    try:
        env_updates = {
            "UI_THEME": ui.theme,
            "UI_LANGUAGE": ui.language,
        }
        _update_env_file(env_updates)
        for key, value in env_updates.items():
            os.environ[key] = value

        return {
            "success": True,
            "message": "UI settings updated successfully",
            "data": {
                "theme": ui.theme,
                "language": ui.language
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test/llm")
async def test_llm_connection():
    """Test LLM connection"""
    from sirchmunk.llm import OpenAIChat

    try:
        base_url = os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)
        api_key = os.getenv("LLM_API_KEY", "")
        model = os.getenv("LLM_MODEL_NAME", _DEFAULT_LLM_MODEL_NAME)

        print(f"[DEBUG] Testing LLM connection with base_url={base_url}, model={model}, api_key={'***' if api_key else '(not set)'}")

        if not api_key:
            return {
                "success": False,
                "status": "error",
                "message": "LLM API key is not configured",
                "model": None
            }

        if not base_url:
            return {
                "success": False,
                "status": "error",
                "message": "LLM base URL is not configured",
                "model": None
            }

        llm = OpenAIChat(
            base_url=base_url,
            api_key=api_key,
            model=model
        )

        messages = [
            {"role": "system",
             "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Output the word: 'test'."}
        ]
        resp = await llm.achat(
            messages=messages,
            stream=False
        )
        print(f"[DEBUG] LLM response: {resp.content}")

        return {
            "success": True,
            "status": "configured",
            "message": "LLM connection successful",
            "model": model,
            "base_url": base_url
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "message": str(e),
            "model": None
        }

@router.get("/status")
async def get_settings_status():
    """Get settings status for quick overview"""
    try:
        ui_settings = get_default_ui_settings()

        llm_api_key = os.getenv("LLM_API_KEY", "")
        llm_base_url = os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)
        llm_model = os.getenv("LLM_MODEL_NAME", _DEFAULT_LLM_MODEL_NAME)

        llm_configured = bool(llm_api_key and llm_base_url and llm_model)

        return {
            "success": True,
            "data": {
                "ui": {
                    "theme": ui_settings.get("theme", "light"),
                    "language": ui_settings.get("language", "en")
                },
                "llm": {
                    "configured": llm_configured,
                    "model": llm_model if llm_configured else None,
                    "status": "ready" if llm_configured else "not_configured"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

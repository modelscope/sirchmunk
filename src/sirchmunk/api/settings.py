# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Settings API endpoints with persistent storage
Provides UI settings and environment variable management

Environment variable resolution priority:
  1. SettingsStorage (DuckDB) - values saved via WebUI
  2. os.getenv() - includes .env file (loaded at app startup) + system env vars
  3. Hardcoded defaults

When users save settings via WebUI, the values are persisted to:
  - SettingsStorage (DuckDB) for immediate use
  - os.environ for current process
  - .env file on disk for persistence across restarts
"""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel

from sirchmunk.api.components.settings_storage import SettingsStorage

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

# Default values (used when neither SettingsStorage, .env, nor os.environ have a value)
_DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LLM_MODEL_NAME = "gpt-5.2"
_DEFAULT_GREP_CONCURRENT_LIMIT = "5"
_DEFAULT_WORK_PATH = os.path.expanduser("~/.sirchmunk")

# Initialize settings storage (with error handling)
try:
    settings_storage = SettingsStorage()
except Exception as e:
    print(f"[ERROR] Failed to initialize SettingsStorage: {e}")
    settings_storage = None


def _get_env_file_path() -> Path:
    """Get the .env file path in the Sirchmunk work directory."""
    work_path = os.getenv("SIRCHMUNK_WORK_PATH", _DEFAULT_WORK_PATH)
    return Path(work_path).expanduser().resolve() / ".env"


def _update_env_file(updates: Dict[str, str]):
    """Update specific key-value pairs in the .env file.

    Preserves comments, blank lines, and overall file structure.
    Only updates existing keys or appends new ones at the end.

    Args:
        updates: Dictionary of key-value pairs to update
    """
    env_path = _get_env_file_path()
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text().splitlines()
        updated_keys: set = set()
        new_lines = []

        for line in lines:
            stripped = line.strip()
            # Preserve blank lines and comments
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

        # Append keys that were not found in the original file
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
    LLM_BASE_URL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: Optional[str] = None
    GREP_CONCURRENT_LIMIT: Optional[int] = None

class SaveSettingsRequest(BaseModel):
    ui: Optional[UISettings] = None
    environment: Optional[Dict[str, str]] = None

# === Helper Functions ===

def get_default_ui_settings() -> Dict[str, Any]:
    """Get default UI settings"""
    return {
        "theme": settings_storage.get_setting("ui.theme", "light"),
        "language": settings_storage.get_setting("ui.language", "en"),
    }

def get_current_env_variables() -> Dict[str, Any]:
    """Get current environment variables with priority resolution.

    Priority: SettingsStorage (WebUI) > os.getenv() (includes .env) > defaults
    """
    # Get saved values from SettingsStorage (highest priority: user WebUI overrides)
    saved_work_path = settings_storage.get_env_variable("SIRCHMUNK_WORK_PATH") if settings_storage else ""
    saved_llm_base_url = settings_storage.get_env_variable("LLM_BASE_URL") if settings_storage else ""
    saved_llm_api_key = settings_storage.get_env_variable("LLM_API_KEY") if settings_storage else ""
    saved_llm_model = settings_storage.get_env_variable("LLM_MODEL_NAME") if settings_storage else ""
    saved_grep_limit = settings_storage.get_env_variable("GREP_CONCURRENT_LIMIT") if settings_storage else ""

    # Resolve with fallback: SettingsStorage > os.getenv() (includes .env) > defaults
    work_path = saved_work_path or os.getenv("SIRCHMUNK_WORK_PATH", _DEFAULT_WORK_PATH)
    llm_base_url = saved_llm_base_url or os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)
    llm_api_key = saved_llm_api_key or os.getenv("LLM_API_KEY", "")
    llm_model = saved_llm_model or os.getenv("LLM_MODEL_NAME", _DEFAULT_LLM_MODEL_NAME)
    grep_limit = saved_grep_limit or os.getenv("GREP_CONCURRENT_LIMIT", _DEFAULT_GREP_CONCURRENT_LIMIT)

    return {
        "SIRCHMUNK_WORK_PATH": {
            "value": work_path,
            "default": _DEFAULT_WORK_PATH,
            "description": "Working directory for Sirchmunk data",
            "category": "system"
        },
        "LLM_BASE_URL": {
            "value": llm_base_url,
            "default": _DEFAULT_LLM_BASE_URL,
            "description": "Base URL for LLM API (OpenAI-compatible endpoint)",
            "category": "llm"
        },
        "LLM_API_KEY": {
            "value": llm_api_key,
            "default": "",
            "description": "API key for LLM service",
            "category": "llm",
            "sensitive": True
        },
        "LLM_MODEL_NAME": {
            "value": llm_model,
            "default": _DEFAULT_LLM_MODEL_NAME,
            "description": "Model name for LLM",
            "category": "llm"
        },
        "GREP_CONCURRENT_LIMIT": {
            "value": grep_limit,
            "default": _DEFAULT_GREP_CONCURRENT_LIMIT,
            "description": "Maximum concurrent grep requests",
            "category": "system"
        }
    }

# === API Endpoints ===

@router.get("")
async def get_all_settings():
    """Get all settings including UI and environment variables"""
    if settings_storage is None:
        raise HTTPException(status_code=503, detail="Settings storage not available")
    
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
    if settings_storage is None:
        # Return default settings if storage not available
        return {
            "success": True,
            "data": {
                    "theme": "light",
                    "language": "en"
                }
            }
    
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
    if settings_storage is None:
        raise HTTPException(status_code=503, detail="Settings storage not available")
    
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
    if settings_storage is None:
        raise HTTPException(status_code=503, detail="Settings storage not available")
    
    try:
        saved_items = []
        
        # Save UI settings
        if request.ui:
            if request.ui.theme:
                settings_storage.save_setting("ui.theme", request.ui.theme, "ui")
                saved_items.append("theme")
            
            if request.ui.language:
                settings_storage.save_setting("ui.language", request.ui.language, "ui")
                saved_items.append("language")
        
        # Save environment variables
        _SENSITIVE_MASK = "***"
        if request.environment:
            env_updates = {}
            for key, value in request.environment.items():
                if value and value != _SENSITIVE_MASK:
                    # Save to SettingsStorage (DuckDB) for immediate use
                    settings_storage.save_env_variable(
                        key,
                        value,
                        description=f"User-configured {key}",
                        category="llm" if "LLM" in key else "system"
                    )
                    # Update os.environ so current process picks up changes
                    os.environ[key] = str(value)
                    # Collect for .env file writeback
                    env_updates[key] = str(value)
                    saved_items.append(key)

            # Persist changes to .env file for next restart
            if env_updates:
                _update_env_file(env_updates)

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
    if settings_storage is None:
        raise HTTPException(status_code=503, detail="Settings storage not available")
    
    try:
        settings_storage.save_setting("ui.theme", ui.theme, "ui")
        settings_storage.save_setting("ui.language", ui.language, "ui")
        
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
        # Get current LLM settings with priority resolution
        base_url = (
            (settings_storage.get_env_variable("LLM_BASE_URL") if settings_storage else "")
            or os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)
        )
        api_key = (
            (settings_storage.get_env_variable("LLM_API_KEY") if settings_storage else "")
            or os.getenv("LLM_API_KEY", "")
        )
        model = (
            (settings_storage.get_env_variable("LLM_MODEL_NAME") if settings_storage else "")
            or os.getenv("LLM_MODEL_NAME", _DEFAULT_LLM_MODEL_NAME)
        )

        print(f"[DEBUG] Testing LLM connection with base_url={base_url}, model={model}, api_key={'***' if api_key else '(not set)'}")
        
        # Simple validation
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
        
        # Check LLM configuration with priority resolution
        llm_api_key = (
            (settings_storage.get_env_variable("LLM_API_KEY") if settings_storage else "")
            or os.getenv("LLM_API_KEY", "")
        )
        llm_base_url = (
            (settings_storage.get_env_variable("LLM_BASE_URL") if settings_storage else "")
            or os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)
        )
        llm_model = (
            (settings_storage.get_env_variable("LLM_MODEL_NAME") if settings_storage else "")
            or os.getenv("LLM_MODEL_NAME", _DEFAULT_LLM_MODEL_NAME)
        )
        
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

# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Settings API endpoints with persistent storage
Provides UI settings and environment variable management
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel
import os

from api.components.settings_storage import SettingsStorage
from sirchmunk.utils.constants import (
    GREP_CONCURRENT_LIMIT,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL_NAME,
    WORK_PATH
)

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

# Initialize settings storage (with error handling)
try:
    settings_storage = SettingsStorage()
except Exception as e:
    print(f"[ERROR] Failed to initialize SettingsStorage: {e}")
    settings_storage = None

# === Request/Response Models ===

class UISettings(BaseModel):
    theme: str = "light"
    language: str = "en"

class EnvironmentVariables(BaseModel):
    WORK_PATH: Optional[str] = None
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
    """Get current environment variables with saved overrides"""
    # Get saved values from storage
    saved_work_path = settings_storage.get_env_variable("WORK_PATH")
    saved_llm_base_url = settings_storage.get_env_variable("LLM_BASE_URL")
    saved_llm_api_key = settings_storage.get_env_variable("LLM_API_KEY")
    saved_llm_model = settings_storage.get_env_variable("LLM_MODEL_NAME")
    saved_grep_limit = settings_storage.get_env_variable("GREP_CONCURRENT_LIMIT")
    
    return {
        "WORK_PATH": {
            "value": saved_work_path or str(WORK_PATH),
            "default": str(WORK_PATH),
            "description": "Working directory for Sirchmunk data",
            "category": "system"
        },
        "LLM_BASE_URL": {
            "value": saved_llm_base_url or LLM_BASE_URL,
            "default": LLM_BASE_URL,
            "description": "Base URL for LLM API (OpenAI-compatible endpoint)",
            "category": "llm"
        },
        "LLM_API_KEY": {
            "value": "***" if (saved_llm_api_key or LLM_API_KEY) else "",
            "default": "***" if LLM_API_KEY else "",
            "description": "API key for LLM service",
            "category": "llm",
                "sensitive": True
            },
        "LLM_MODEL_NAME": {
            "value": saved_llm_model or LLM_MODEL_NAME,
            "default": LLM_MODEL_NAME,
            "description": "Model name for LLM",
            "category": "llm"
        },
        "GREP_CONCURRENT_LIMIT": {
            "value": saved_grep_limit or str(GREP_CONCURRENT_LIMIT),
            "default": str(GREP_CONCURRENT_LIMIT),
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
        if request.environment:
            for key, value in request.environment.items():
                if value:  # Only save non-empty values
                    # For sensitive fields, store the actual value (not masked)
                    settings_storage.save_env_variable(
                        key, 
                        value,
                        description=f"User-configured {key}",
                        category="llm" if "LLM" in key else "system"
                    )
                    saved_items.append(key)
    
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
        # Get current LLM settings
        base_url = settings_storage.get_env_variable("LLM_BASE_URL") or LLM_BASE_URL
        api_key = settings_storage.get_env_variable("LLM_API_KEY") or LLM_API_KEY
        model = settings_storage.get_env_variable("LLM_MODEL_NAME") or LLM_MODEL_NAME

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
        print(f"[DEBUG] LLM response: {resp}")

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
        
        # Check LLM configuration
        llm_api_key = settings_storage.get_env_variable("LLM_API_KEY") or LLM_API_KEY
        llm_base_url = settings_storage.get_env_variable("LLM_BASE_URL") or LLM_BASE_URL
        llm_model = settings_storage.get_env_variable("LLM_MODEL_NAME") or LLM_MODEL_NAME
        
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

# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Mock API endpoints for system settings and configuration
Provides system status, configuration management, and health monitoring
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime
import random

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

# Mock system configuration
system_config = {
    "ui": {
        "language": "en",
        "theme": "light",
        "sidebar_width": 280,
        "sidebar_collapsed": False,
        "notifications_enabled": True,
        "auto_save_interval": 30,
        "max_chat_history": 100
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 30
    },
    "embedding": {
        "provider": "openai", 
        "model": "text-embedding-ada-002",
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "tts": {
        "enabled": True,
        "provider": "azure",
        "voice": "en-US-AriaNeural",
        "speed": 1.0,
        "pitch": 0
    }
}

# Mock LLM providers and models
llm_providers = [
    {
        "name": "openai",
        "display_name": "OpenAI",
        "description": "OpenAI GPT models",
        "models": [
            {"id": "gpt-4", "name": "GPT-4", "context_length": 8192},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "context_length": 128000},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context_length": 4096}
        ],
        "config_fields": [
            {"name": "api_key", "type": "password", "required": True, "description": "OpenAI API Key"},
            {"name": "organization", "type": "text", "required": False, "description": "Organization ID"}
        ],
        "active": True
    },
    {
        "name": "anthropic",
        "display_name": "Anthropic",
        "description": "Claude models by Anthropic",
        "models": [
            {"id": "claude-3-opus", "name": "Claude 3 Opus", "context_length": 200000},
            {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "context_length": 200000},
            {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "context_length": 200000}
        ],
        "config_fields": [
            {"name": "api_key", "type": "password", "required": True, "description": "Anthropic API Key"}
        ],
        "active": False
    },
    {
        "name": "azure",
        "display_name": "Azure OpenAI",
        "description": "Azure OpenAI Service",
        "models": [
            {"id": "gpt-4", "name": "GPT-4", "context_length": 8192},
            {"id": "gpt-35-turbo", "name": "GPT-3.5 Turbo", "context_length": 4096}
        ],
        "config_fields": [
            {"name": "api_key", "type": "password", "required": True, "description": "Azure API Key"},
            {"name": "endpoint", "type": "text", "required": True, "description": "Azure Endpoint URL"},
            {"name": "api_version", "type": "text", "required": True, "description": "API Version"}
        ],
        "active": False
    }
]

# Mock environment variables configuration
env_config = {
    "llm": {
        "name": "LLM Configuration",
        "description": "Large Language Model settings",
        "icon": "brain",
        "variables": [
            {
                "key": "OPENAI_API_KEY",
                "name": "OpenAI API Key",
                "type": "password",
                "required": True,
                "description": "API key for OpenAI services",
                "value": "sk-..." if random.random() > 0.5 else "",
                "sensitive": True
            },
            {
                "key": "OPENAI_ORGANIZATION",
                "name": "OpenAI Organization",
                "type": "text",
                "required": False,
                "description": "OpenAI organization ID",
                "value": "org-...",
                "sensitive": False
            }
        ]
    },
    "embedding": {
        "name": "Embedding Configuration", 
        "description": "Text embedding service settings",
        "icon": "vector",
        "variables": [
            {
                "key": "EMBEDDING_PROVIDER",
                "name": "Embedding Provider",
                "type": "select",
                "options": ["openai", "huggingface", "sentence-transformers"],
                "required": True,
                "description": "Provider for text embeddings",
                "value": "openai",
                "sensitive": False
            },
            {
                "key": "EMBEDDING_MODEL",
                "name": "Embedding Model",
                "type": "text",
                "required": True,
                "description": "Model name for embeddings",
                "value": "text-embedding-ada-002",
                "sensitive": False
            }
        ]
    },
    "tts": {
        "name": "Text-to-Speech",
        "description": "Speech synthesis configuration",
        "icon": "volume",
        "variables": [
            {
                "key": "TTS_PROVIDER",
                "name": "TTS Provider",
                "type": "select",
                "options": ["azure", "google", "aws", "elevenlabs"],
                "required": True,
                "description": "Text-to-speech service provider",
                "value": "azure",
                "sensitive": False
            },
            {
                "key": "AZURE_SPEECH_KEY",
                "name": "Azure Speech Key",
                "type": "password",
                "required": True,
                "description": "Azure Cognitive Services Speech key",
                "value": "..." if random.random() > 0.3 else "",
                "sensitive": True
            }
        ]
    }
}

# Mock system status
system_status = {
    "backend": {
        "status": "healthy",
        "uptime": "2d 14h 32m",
        "version": "1.0.0",
        "last_check": datetime.now().isoformat() + "Z"
    },
    "database": {
        "status": "healthy",
        "connection_pool": "8/10 active",
        "last_backup": "2024-01-13T02:00:00Z",
        "last_check": datetime.now().isoformat() + "Z"
    },
    "llm_service": {
        "status": random.choice(["healthy", "warning", "error"]),
        "provider": "openai",
        "model": "gpt-4",
        "response_time": f"{random.randint(800, 2000)}ms",
        "last_check": datetime.now().isoformat() + "Z"
    },
    "embedding_service": {
        "status": random.choice(["healthy", "warning"]),
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "response_time": f"{random.randint(200, 800)}ms",
        "last_check": datetime.now().isoformat() + "Z"
    },
    "tts_service": {
        "status": random.choice(["healthy", "disabled"]),
        "provider": "azure",
        "voice": "en-US-AriaNeural",
        "response_time": f"{random.randint(500, 1500)}ms",
        "last_check": datetime.now().isoformat() + "Z"
    }
}

@router.get("/")
async def get_all_settings():
    """Get all system settings"""
    return {
        "success": True,
        "data": {
            "ui": system_config["ui"],
            "llm": system_config["llm"],
            "embedding": system_config["embedding"],
            "tts": system_config["tts"],
            "system_info": {
                "version": "1.0.0",
                "build": "2024.01.13",
                "environment": "development"
            }
        }
    }

@router.put("/")
async def update_settings(request: Dict[str, Any]):
    """Update system settings"""
    global system_config
    
    # Update UI settings
    if "ui" in request:
        ui_settings = request["ui"]
        for key, value in ui_settings.items():
            if key in system_config["ui"]:
                system_config["ui"][key] = value
    
    # Update LLM settings
    if "llm" in request:
        llm_settings = request["llm"]
        for key, value in llm_settings.items():
            if key in system_config["llm"]:
                system_config["llm"][key] = value
    
    # Update other settings
    for section in ["embedding", "tts"]:
        if section in request:
            section_settings = request[section]
            for key, value in section_settings.items():
                if key in system_config[section]:
                    system_config[section][key] = value
    
    return {
        "success": True,
        "data": system_config,
        "message": "Settings updated successfully"
    }

@router.get("/llm/providers")
async def get_llm_providers():
    """Get available LLM providers"""
    return {
        "success": True,
        "data": llm_providers
    }

@router.get("/llm/mode")
async def get_llm_mode():
    """Get current LLM mode and configuration"""
    active_provider = next((p for p in llm_providers if p["active"]), llm_providers[0])
    
    return {
        "success": True,
        "data": {
            "current_provider": active_provider["name"],
            "current_model": system_config["llm"]["model"],
            "available_providers": [p["name"] for p in llm_providers],
            "provider_details": active_provider
        }
    }

@router.post("/llm/providers")
async def create_llm_provider(request: Dict[str, Any]):
    """Create or update LLM provider configuration"""
    provider_name = request.get("name")
    config = request.get("config", {})
    
    if not provider_name:
        raise HTTPException(status_code=400, detail="Provider name is required")
    
    # Find existing provider or create new one
    provider_index = next((i for i, p in enumerate(llm_providers) if p["name"] == provider_name), None)
    
    if provider_index is not None:
        # Update existing provider
        llm_providers[provider_index]["config"] = config
        llm_providers[provider_index]["active"] = request.get("active", False)
        updated_provider = llm_providers[provider_index]
    else:
        # Create new provider (mock - in real implementation would validate)
        new_provider = {
            "name": provider_name,
            "display_name": request.get("display_name", provider_name),
            "description": request.get("description", ""),
            "models": request.get("models", []),
            "config": config,
            "active": request.get("active", False)
        }
        llm_providers.append(new_provider)
        updated_provider = new_provider
    
    return {
        "success": True,
        "data": updated_provider,
        "message": f"Provider '{provider_name}' configured successfully"
    }

@router.delete("/llm/providers/{provider_name}")
async def delete_llm_provider(provider_name: str):
    """Delete LLM provider configuration"""
    global llm_providers
    
    provider_index = next((i for i, p in enumerate(llm_providers) if p["name"] == provider_name), None)
    
    if provider_index is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    # Don't allow deletion of active provider
    if llm_providers[provider_index]["active"]:
        raise HTTPException(status_code=400, detail="Cannot delete active provider")
    
    deleted_provider = llm_providers.pop(provider_index)
    
    return {
        "success": True,
        "message": f"Provider '{provider_name}' deleted successfully",
        "data": deleted_provider
    }

@router.post("/llm/providers/{provider_name}/activate")
async def activate_llm_provider(provider_name: str):
    """Activate LLM provider"""
    global llm_providers
    
    provider = next((p for p in llm_providers if p["name"] == provider_name), None)
    
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    # Deactivate all providers
    for p in llm_providers:
        p["active"] = False
    
    # Activate selected provider
    provider["active"] = True
    
    # Update system config
    system_config["llm"]["provider"] = provider_name
    
    return {
        "success": True,
        "data": provider,
        "message": f"Provider '{provider_name}' activated successfully"
    }

@router.post("/llm/providers/{provider_name}/test")
async def test_llm_provider(provider_name: str):
    """Test LLM provider connection"""
    provider = next((p for p in llm_providers if p["name"] == provider_name), None)
    
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    # Simulate testing delay
    import asyncio
    await asyncio.sleep(random.uniform(1, 3))
    
    # Mock test results
    test_success = random.random() > 0.2  # 80% success rate
    
    if test_success:
        return {
            "success": True,
            "data": {
                "status": "success",
                "response_time": random.randint(500, 2000),
                "model_info": {
                    "name": provider.get("models", [{}])[0].get("name", "Unknown"),
                    "context_length": provider.get("models", [{}])[0].get("context_length", 4096)
                },
                "test_message": "Connection successful! Provider is working correctly."
            }
        }
    else:
        return {
            "success": False,
            "error": {
                "code": "CONNECTION_FAILED",
                "message": "Failed to connect to provider. Please check your configuration.",
                "details": random.choice([
                    "Invalid API key",
                    "Network timeout",
                    "Rate limit exceeded",
                    "Service unavailable"
                ])
            }
        }

@router.get("/env")
async def get_env_config():
    """Get environment configuration"""
    return {
        "success": True,
        "data": env_config
    }

@router.put("/env")
async def update_env_config(request: Dict[str, Any]):
    """Update environment variables"""
    global env_config
    
    updated_vars = []
    
    for category, variables in request.items():
        if category in env_config:
            for var_key, var_value in variables.items():
                # Find and update variable
                for var in env_config[category]["variables"]:
                    if var["key"] == var_key:
                        var["value"] = var_value
                        updated_vars.append(var_key)
                        break
    
    return {
        "success": True,
        "data": {
            "updated_variables": updated_vars,
            "config": env_config
        },
        "message": f"Updated {len(updated_vars)} environment variables"
    }

@router.post("/env/test")
async def test_env_config():
    """Test environment configuration"""
    # Simulate testing delay
    import asyncio
    await asyncio.sleep(random.uniform(2, 4))
    
    # Mock test results for each service
    test_results = {}
    
    for service in ["llm", "embedding", "tts"]:
        success = random.random() > 0.15  # 85% success rate
        
        test_results[service] = {
            "status": "success" if success else "error",
            "response_time": random.randint(200, 2000) if success else None,
            "message": "Service is working correctly" if success else "Configuration error",
            "details": None if success else random.choice([
                "Invalid API key",
                "Service unavailable", 
                "Network timeout",
                "Authentication failed"
            ])
        }
    
    overall_success = all(result["status"] == "success" for result in test_results.values())
    
    return {
        "success": overall_success,
        "data": {
            "overall_status": "success" if overall_success else "partial_failure",
            "services": test_results,
            "tested_at": datetime.now().isoformat() + "Z"
        }
    }

@router.get("/status")
async def get_system_status():
    """Get system health status"""
    # Update status with current timestamp
    for service in system_status.values():
        service["last_check"] = datetime.now().isoformat() + "Z"
    
    # Calculate overall health
    service_statuses = [s["status"] for s in system_status.values()]
    if all(status == "healthy" for status in service_statuses):
        overall_status = "healthy"
    elif any(status == "error" for status in service_statuses):
        overall_status = "error"
    else:
        overall_status = "warning"
    
    return {
        "success": True,
        "data": {
            "overall_status": overall_status,
            "services": system_status,
            "last_updated": datetime.now().isoformat() + "Z"
        }
    }

@router.post("/status/test/{service}")
async def test_service(service: str):
    """Test specific service"""
    if service not in system_status:
        raise HTTPException(status_code=404, detail="Service not found")
    
    # Simulate testing delay
    import asyncio
    await asyncio.sleep(random.uniform(1, 3))
    
    # Mock test result
    success = random.random() > 0.1  # 90% success rate
    
    # Update service status
    system_status[service]["status"] = "healthy" if success else "error"
    system_status[service]["last_check"] = datetime.now().isoformat() + "Z"
    
    if success:
        system_status[service]["response_time"] = f"{random.randint(200, 2000)}ms"
    
    return {
        "success": success,
        "data": {
            "service": service,
            "status": system_status[service]["status"],
            "response_time": system_status[service].get("response_time"),
            "message": "Service test completed successfully" if success else "Service test failed"
        }
    }

@router.get("/backup")
async def get_backup_info():
    """Get backup information"""
    return {
        "success": True,
        "data": {
            "last_backup": "2024-01-13T02:00:00Z",
            "backup_size": "2.3 GB",
            "backup_location": "/backups/deeptutor_20240113.sql",
            "auto_backup_enabled": True,
            "backup_schedule": "Daily at 2:00 AM",
            "retention_days": 30
        }
    }

@router.post("/backup/create")
async def create_backup():
    """Create system backup"""
    # Simulate backup creation
    import asyncio
    await asyncio.sleep(random.uniform(3, 6))
    
    backup_id = str(uuid.uuid4())
    
    return {
        "success": True,
        "data": {
            "backup_id": backup_id,
            "created_at": datetime.now().isoformat() + "Z",
            "size": f"{random.uniform(1.5, 3.0):.1f} GB",
            "location": f"/backups/deeptutor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        },
        "message": "Backup created successfully"
    }
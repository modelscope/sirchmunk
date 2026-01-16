# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Mock API endpoints for system monitoring functionality
Provides real-time monitoring of tasks, files, tokens, and system metrics
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime, timedelta
import random
import psutil
import os

router = APIRouter(prefix="/api/v1/monitor", tags=["monitor"])

# Mock data storage
active_tasks = {}
processed_files = {}
token_usage_stats = {}
system_metrics_history = []

def generate_mock_task_data():
    """Generate mock task data"""
    tasks = [
        {
            "id": f"task_{i:03d}",
            "type": random.choice(["solve", "research", "chat", "upload"]),
            "name": random.choice([
                "Machine Learning Problem Analysis",
                "Climate Change Research Report", 
                "AI Assistant Conversation",
                "Document Processing",
                "Data Analysis Pipeline",
                "Natural Language Processing",
                "Computer Vision Task",
                "Knowledge Base Update"
            ]),
            "status": random.choice(["running", "completed", "failed", "pending"]),
            "progress": random.randint(0, 100) if random.choice([True, False]) else 100,
            "startTime": (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat(),
            "endTime": (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat() if random.choice([True, False]) else None,
            "duration": random.randint(30, 1800),
            "tokensUsed": random.randint(100, 5000),
            "filesProcessed": random.randint(0, 10),
            "error": random.choice([None, "Connection timeout", "File format not supported", "Insufficient memory"]) if random.random() < 0.2 else None
        }
        for i in range(random.randint(5, 15))
    ]
    return tasks

def generate_mock_file_data():
    """Generate mock file data"""
    file_types = ["PDF", "DOCX", "TXT", "CSV", "JSON", "XLSX"]
    file_names = [
        "research_paper.pdf", "dataset.csv", "presentation.pptx", "report.docx",
        "analysis.xlsx", "config.json", "notes.txt", "data_export.csv",
        "whitepaper.pdf", "specifications.docx"
    ]
    
    files = [
        {
            "id": f"file_{i:03d}",
            "name": random.choice(file_names),
            "type": random.choice(file_types),
            "size": random.randint(1024, 10485760),  # 1KB to 10MB
            "uploadTime": (datetime.now() - timedelta(hours=random.randint(1, 168))).isoformat(),
            "processedTime": (datetime.now() - timedelta(hours=random.randint(0, 24))).isoformat() if random.choice([True, False]) else None,
            "status": random.choice(["uploaded", "processing", "processed", "failed"]),
            "tokensGenerated": random.randint(0, 2000),
            "associatedTasks": [f"task_{random.randint(1, 10):03d}" for _ in range(random.randint(0, 3))]
        }
        for i in range(random.randint(8, 20))
    ]
    return files

def generate_mock_token_usage():
    """Generate mock token usage data"""
    models = [
        "gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002", 
        "gpt-4-turbo", "claude-3-opus", "claude-3-sonnet"
    ]
    
    usage_data = []
    for model in models:
        if random.choice([True, False]):  # Not all models may be used
            total_tokens = random.randint(1000, 50000)
            input_tokens = int(total_tokens * random.uniform(0.4, 0.7))
            output_tokens = total_tokens - input_tokens
            
            # Mock pricing (approximate)
            cost_per_1k = {
                "gpt-4": 0.03,
                "gpt-3.5-turbo": 0.002,
                "text-embedding-ada-002": 0.0001,
                "gpt-4-turbo": 0.01,
                "claude-3-opus": 0.015,
                "claude-3-sonnet": 0.003
            }
            
            cost = (total_tokens / 1000) * cost_per_1k.get(model, 0.01)
            
            usage_data.append({
                "model": model,
                "totalTokens": total_tokens,
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "cost": round(cost, 4),
                "requests": random.randint(10, 200),
                "lastUsed": (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat()
            })
    
    return usage_data

def get_system_metrics():
    """Get real system metrics where possible, mock where not available"""
    try:
        # Try to get real system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get network connections (approximate)
        connections = len(psutil.net_connections())
        
        # Get system uptime
        boot_time = psutil.boot_time()
        uptime_seconds = datetime.now().timestamp() - boot_time
        uptime_days = int(uptime_seconds // 86400)
        uptime_hours = int((uptime_seconds % 86400) // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{uptime_days}d {uptime_hours}h {uptime_minutes}m"
        
        return {
            "cpuUsage": round(cpu_percent, 1),
            "memoryUsage": round(memory.percent, 1),
            "diskUsage": round(disk.percent, 1),
            "activeConnections": min(connections, 50),  # Cap for display
            "uptime": uptime_str,
            "lastUpdated": datetime.now().isoformat()
        }
    except:
        # Fallback to mock data if psutil is not available or fails
        return {
            "cpuUsage": round(random.uniform(20, 80), 1),
            "memoryUsage": round(random.uniform(40, 90), 1),
            "diskUsage": round(random.uniform(20, 60), 1),
            "activeConnections": random.randint(5, 25),
            "uptime": f"{random.randint(0, 30)}d {random.randint(0, 23)}h {random.randint(0, 59)}m",
            "lastUpdated": datetime.now().isoformat()
        }

@router.get("/tasks")
async def get_tasks(status: Optional[str] = None, limit: int = 50):
    """Get current tasks with optional status filter"""
    tasks = generate_mock_task_data()
    
    if status and status != "all":
        tasks = [task for task in tasks if task["status"] == status]
    
    # Sort by start time (most recent first)
    tasks.sort(key=lambda x: x["startTime"], reverse=True)
    
    return {
        "success": True,
        "data": tasks[:limit],
        "total": len(tasks),
        "summary": {
            "running": len([t for t in tasks if t["status"] == "running"]),
            "completed": len([t for t in tasks if t["status"] == "completed"]),
            "failed": len([t for t in tasks if t["status"] == "failed"]),
            "pending": len([t for t in tasks if t["status"] == "pending"])
        }
    }

@router.get("/files")
async def get_files(status: Optional[str] = None, limit: int = 50):
    """Get processed files with optional status filter"""
    files = generate_mock_file_data()
    
    if status and status != "all":
        files = [file for file in files if file["status"] == status]
    
    # Sort by upload time (most recent first)
    files.sort(key=lambda x: x["uploadTime"], reverse=True)
    
    return {
        "success": True,
        "data": files[:limit],
        "total": len(files),
        "summary": {
            "uploaded": len([f for f in files if f["status"] == "uploaded"]),
            "processing": len([f for f in files if f["status"] == "processing"]),
            "processed": len([f for f in files if f["status"] == "processed"]),
            "failed": len([f for f in files if f["status"] == "failed"]),
            "totalSize": sum(f["size"] for f in files),
            "totalTokens": sum(f["tokensGenerated"] for f in files)
        }
    }

@router.get("/tokens")
async def get_token_usage():
    """Get token usage statistics by model"""
    usage_data = generate_mock_token_usage()
    
    total_tokens = sum(u["totalTokens"] for u in usage_data)
    total_cost = sum(u["cost"] for u in usage_data)
    total_requests = sum(u["requests"] for u in usage_data)
    
    return {
        "success": True,
        "data": usage_data,
        "summary": {
            "totalTokens": total_tokens,
            "totalCost": round(total_cost, 4),
            "totalRequests": total_requests,
            "modelsUsed": len(usage_data),
            "averageCostPerToken": round(total_cost / total_tokens * 1000, 6) if total_tokens > 0 else 0
        }
    }

@router.get("/system")
async def get_system_status():
    """Get current system metrics and status"""
    metrics = get_system_metrics()
    
    # Add some additional computed metrics
    health_score = 100
    if metrics["cpuUsage"] > 80:
        health_score -= 20
    if metrics["memoryUsage"] > 85:
        health_score -= 25
    if metrics["diskUsage"] > 90:
        health_score -= 30
    
    health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "warning" if health_score >= 50 else "critical"
    
    return {
        "success": True,
        "data": {
            **metrics,
            "healthScore": max(0, health_score),
            "healthStatus": health_status,
            "services": {
                "api": "running",
                "database": "connected",
                "cache": "active",
                "queue": "processing"
            }
        }
    }

@router.get("/overview")
async def get_monitoring_overview():
    """Get comprehensive monitoring overview"""
    tasks = generate_mock_task_data()
    files = generate_mock_file_data()
    tokens = generate_mock_token_usage()
    system = get_system_metrics()
    
    # Calculate summary statistics
    active_tasks_count = len([t for t in tasks if t["status"] == "running"])
    completed_tasks_count = len([t for t in tasks if t["status"] == "completed"])
    total_tokens = sum(u["totalTokens"] for u in tokens)
    total_cost = sum(u["cost"] for u in tokens)
    processed_files_count = len([f for f in files if f["status"] == "processed"])
    
    return {
        "success": True,
        "data": {
            "tasks": {
                "active": active_tasks_count,
                "completed": completed_tasks_count,
                "total": len(tasks)
            },
            "files": {
                "processed": processed_files_count,
                "total": len(files),
                "totalSize": sum(f["size"] for f in files)
            },
            "tokens": {
                "total": total_tokens,
                "cost": round(total_cost, 4),
                "models": len(tokens)
            },
            "system": {
                "cpu": system["cpuUsage"],
                "memory": system["memoryUsage"],
                "disk": system["diskUsage"],
                "uptime": system["uptime"]
            },
            "lastUpdated": datetime.now().isoformat()
        }
    }

@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    # In a real implementation, this would cancel the actual task
    return {
        "success": True,
        "message": f"Task {task_id} cancellation requested",
        "data": {
            "taskId": task_id,
            "status": "cancelling",
            "timestamp": datetime.now().isoformat()
        }
    }

@router.post("/tasks/{task_id}/retry")
async def retry_task(task_id: str):
    """Retry a failed task"""
    # In a real implementation, this would restart the failed task
    return {
        "success": True,
        "message": f"Task {task_id} retry initiated",
        "data": {
            "taskId": task_id,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
    }

@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a processed file"""
    # In a real implementation, this would delete the actual file
    return {
        "success": True,
        "message": f"File {file_id} deleted successfully",
        "data": {
            "fileId": file_id,
            "timestamp": datetime.now().isoformat()
        }
    }

@router.post("/system/cleanup")
async def cleanup_system():
    """Perform system cleanup operations"""
    # In a real implementation, this would perform actual cleanup
    cleanup_results = {
        "tempFilesRemoved": random.randint(10, 100),
        "cacheCleared": f"{random.randint(50, 500)}MB",
        "logsRotated": random.randint(5, 20),
        "memoryFreed": f"{random.randint(100, 1000)}MB"
    }
    
    return {
        "success": True,
        "message": "System cleanup completed",
        "data": {
            "results": cleanup_results,
            "timestamp": datetime.now().isoformat()
        }
    }

@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    system_metrics = get_system_metrics()
    
    # Check various system components
    checks = {
        "api": {"status": "healthy", "responseTime": f"{random.randint(10, 50)}ms"},
        "database": {"status": "healthy", "connections": random.randint(5, 20)},
        "cache": {"status": "healthy", "hitRate": f"{random.randint(85, 98)}%"},
        "storage": {"status": "healthy" if system_metrics["diskUsage"] < 90 else "warning", "usage": f"{system_metrics['diskUsage']}%"},
        "memory": {"status": "healthy" if system_metrics["memoryUsage"] < 85 else "warning", "usage": f"{system_metrics['memoryUsage']}%"},
        "cpu": {"status": "healthy" if system_metrics["cpuUsage"] < 80 else "warning", "usage": f"{system_metrics['cpuUsage']}%"}
    }
    
    overall_status = "healthy"
    if any(check["status"] == "warning" for check in checks.values()):
        overall_status = "warning"
    if any(check["status"] == "error" for check in checks.values()):
        overall_status = "error"
    
    return {
        "success": True,
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "uptime": system_metrics["uptime"]
    }
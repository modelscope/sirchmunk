# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Real-time system monitoring API endpoints
Provides actual system metrics and activity tracking
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime

from api.components.monitor_tracker import get_monitor_tracker

router = APIRouter(prefix="/api/v1/monitor", tags=["monitor"])

# === API Endpoints ===

@router.get("/overview")
async def get_monitoring_overview():
    """
    Get comprehensive monitoring overview
    
    Returns:
        - System metrics (CPU, memory, disk)
        - Chat activity statistics
        - Knowledge cluster statistics
        - Storage information
        - Health status
    """
    try:
        tracker = get_monitor_tracker()
        overview = tracker.get_overview()
        
        return {
            "success": True,
            "data": overview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system")
async def get_system_metrics():
    """
    Get current system metrics
    
    Returns:
        - CPU usage and count
        - Memory usage and capacity
        - Disk usage and capacity
        - Network connections
        - System uptime
        - Process-specific metrics
    """
    try:
        tracker = get_monitor_tracker()
        metrics = tracker.get_system_metrics()
        
        return {
            "success": True,
            "data": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_health_status():
    """
    Get comprehensive health status
    
    Returns:
        - Overall health score
        - Status (excellent/good/warning/critical)
        - Issues list
        - Service availability
    """
    try:
        tracker = get_monitor_tracker()
        health = tracker.get_health_status()
        
        return {
            "success": True,
            "data": health
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat")
async def get_chat_activity(hours: int = 24):
    """
    Get chat activity statistics
    
    Query params:
        hours: Time window in hours (default: 24)
    
    Returns:
        - Total sessions
        - Total messages
        - Recent sessions (top 10)
        - Active sessions count
    """
    try:
        tracker = get_monitor_tracker()
        activity = tracker.get_chat_activity(hours=hours)
        
        return {
            "success": True,
            "data": activity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge")
async def get_knowledge_activity():
    """
    Get knowledge cluster activity statistics
    
    Returns:
        - Total clusters
        - Recent clusters (top 10)
        - Lifecycle distribution
        - Average confidence
    """
    try:
        tracker = get_monitor_tracker()
        activity = tracker.get_knowledge_activity()
        
        return {
            "success": True,
            "data": activity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage")
async def get_storage_info():
    """
    Get storage information
    
    Returns:
        - Work path
        - Cache path
        - Database sizes
        - Total cache size
    """
    try:
        tracker = get_monitor_tracker()
        storage = tracker.get_storage_info()
        
        return {
            "success": True,
            "data": storage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_simple_status():
    """
    Get simple status summary (for quick health checks)
    
    Returns:
        Basic system status information
    """
    try:
        tracker = get_monitor_tracker()
        health = tracker.get_health_status()
        metrics = tracker.get_system_metrics()
        
        return {
            "success": True,
            "status": health["overall_status"],
            "health_score": health["health_score"],
            "cpu_usage": metrics.get("cpu", {}).get("usage_percent", 0),
            "memory_usage": metrics.get("memory", {}).get("usage_percent", 0),
            "disk_usage": metrics.get("disk", {}).get("usage_percent", 0),
            "uptime": metrics.get("uptime", ""),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refresh")
async def refresh_metrics():
    """
    Force refresh of monitoring metrics
    
    Returns:
        Updated overview data
    """
    try:
        # Simply get fresh data
        tracker = get_monitor_tracker()
        overview = tracker.get_overview()
        
        return {
            "success": True,
            "message": "Metrics refreshed successfully",
            "data": overview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

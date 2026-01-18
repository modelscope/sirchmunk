# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Real-time monitoring and tracking component
Provides actual system metrics and activity tracking
"""

import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from sirchmunk.storage.knowledge_manager import KnowledgeManager
from api.components.history_storage import HistoryStorage


class MonitorTracker:
    """
    Real-time system monitoring and activity tracking
    
    Architecture:
    - Tracks actual chat sessions
    - Monitors knowledge cluster creation
    - Collects real system metrics
    - Provides comprehensive statistics
    """
    
    def __init__(self):
        """Initialize monitoring components"""
        try:
            self.history_storage = HistoryStorage()
        except:
            self.history_storage = None
        
        try:
            self.knowledge_manager = KnowledgeManager()
        except:
            self.knowledge_manager = None
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get real system metrics
        
        Returns:
            Dictionary with CPU, memory, disk, and network metrics
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024 ** 3)
            memory_used_gb = memory.used / (1024 ** 3)
            memory_available_gb = memory.available / (1024 ** 3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_total_gb = disk.total / (1024 ** 3)
            disk_used_gb = disk.used / (1024 ** 3)
            disk_free_gb = disk.free / (1024 ** 3)
            
            # Network connections (limit to reasonable number for display)
            try:
                connections = len(psutil.net_connections())
            except:
                connections = 0
            
            # System uptime
            boot_time = psutil.boot_time()
            uptime_seconds = datetime.now().timestamp() - boot_time
            uptime_days = int(uptime_seconds // 86400)
            uptime_hours = int((uptime_seconds % 86400) // 3600)
            uptime_minutes = int((uptime_seconds % 3600) // 60)
            uptime_str = f"{uptime_days}d {uptime_hours}h {uptime_minutes}m"
            
            # Process info
            process = psutil.Process(os.getpid())
            process_memory_mb = process.memory_info().rss / (1024 ** 2)
            process_cpu_percent = process.cpu_percent(interval=0.1)
            
            return {
                "cpu": {
                    "usage_percent": round(cpu_percent, 1),
                    "count": cpu_count,
                    "process_percent": round(process_cpu_percent, 1),
                },
                "memory": {
                    "usage_percent": round(memory.percent, 1),
                    "total_gb": round(memory_total_gb, 2),
                    "used_gb": round(memory_used_gb, 2),
                    "available_gb": round(memory_available_gb, 2),
                    "process_mb": round(process_memory_mb, 1),
                },
                "disk": {
                    "usage_percent": round(disk.percent, 1),
                    "total_gb": round(disk_total_gb, 2),
                    "used_gb": round(disk_used_gb, 2),
                    "free_gb": round(disk_free_gb, 2),
                },
                "network": {
                    "active_connections": connections,
                },
                "uptime": uptime_str,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            # Minimal fallback
            return {
                "cpu": {"usage_percent": 0, "count": 1, "process_percent": 0},
                "memory": {"usage_percent": 0, "total_gb": 0, "used_gb": 0, "available_gb": 0, "process_mb": 0},
                "disk": {"usage_percent": 0, "total_gb": 0, "used_gb": 0, "free_gb": 0},
                "network": {"active_connections": 0},
                "uptime": "0d 0h 0m",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def get_chat_activity(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get chat activity statistics
        
        Args:
            hours: Time window in hours
        
        Returns:
            Chat activity statistics
        """
        if not self.history_storage:
            return {
                "total_sessions": 0,
                "total_messages": 0,
                "recent_sessions": [],
                "active_sessions": 0,
            }
        
        try:
            # Get all sessions
            all_sessions = self.history_storage.get_all_sessions()
            
            # Calculate time threshold
            threshold = datetime.now() - timedelta(hours=hours)
            threshold_ts = threshold.timestamp()
            
            # Filter recent sessions
            recent_sessions = []
            total_messages = 0
            active_count = 0
            
            for session in all_sessions:
                session_time = session.get('updated_at', 0)
                
                # Count messages
                messages = session.get('messages', [])
                total_messages += len(messages)
                
                # Check if recent
                if session_time >= threshold_ts:
                    recent_sessions.append({
                        "session_id": session.get('session_id'),
                        "title": session.get('title', 'Untitled'),
                        "message_count": len(messages),
                        "created_at": session.get('created_at'),
                        "updated_at": session.get('updated_at'),
                    })
                    active_count += 1
            
            # Sort by update time
            recent_sessions.sort(key=lambda x: x['updated_at'], reverse=True)
            
            return {
                "total_sessions": len(all_sessions),
                "total_messages": total_messages,
                "recent_sessions": recent_sessions[:10],  # Top 10 most recent
                "active_sessions": active_count,
                "time_window_hours": hours,
            }
        
        except Exception as e:
            return {
                "total_sessions": 0,
                "total_messages": 0,
                "recent_sessions": [],
                "active_sessions": 0,
                "error": str(e)
            }
    
    def get_knowledge_activity(self) -> Dict[str, Any]:
        """
        Get knowledge cluster activity statistics
        
        Returns:
            Knowledge cluster statistics
        """
        if not self.knowledge_manager:
            return {
                "total_clusters": 0,
                "recent_clusters": [],
                "lifecycle_distribution": {},
            }
        
        try:
            stats = self.knowledge_manager.get_stats()
            custom_stats = stats.get('custom_stats', {})
            
            # Get recent clusters
            recent_rows = self.knowledge_manager.db.fetch_all(
                """
                SELECT id, name, lifecycle, last_modified, confidence
                FROM knowledge_clusters
                ORDER BY last_modified DESC
                LIMIT 10
                """
            )
            
            recent_clusters = [
                {
                    "id": row[0],
                    "name": row[1],
                    "lifecycle": row[2],
                    "last_modified": row[3],
                    "confidence": row[4],
                }
                for row in recent_rows
            ]
            
            return {
                "total_clusters": custom_stats.get('total_clusters', 0),
                "recent_clusters": recent_clusters,
                "lifecycle_distribution": custom_stats.get('lifecycle_distribution', {}),
                "average_confidence": custom_stats.get('average_confidence', 0),
            }
        
        except Exception as e:
            return {
                "total_clusters": 0,
                "recent_clusters": [],
                "lifecycle_distribution": {},
                "error": str(e)
            }
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information for databases and cache
        
        Returns:
            Storage information
        """
        try:
            work_path = Path(os.getenv("WORK_PATH", os.path.expanduser("~/sirchmunk")))
            cache_path = work_path / ".cache"
            
            storage_info = {
                "work_path": str(work_path),
                "cache_path": str(cache_path),
                "databases": {},
            }
            
            # Check history database
            history_db = cache_path / "history" / "chat_history.db"
            if history_db.exists():
                size_mb = history_db.stat().st_size / (1024 ** 2)
                storage_info["databases"]["history"] = {
                    "path": str(history_db),
                    "size_mb": round(size_mb, 2),
                    "exists": True,
                }
            
            # Check knowledge parquet
            knowledge_parquet = cache_path / "knowledge" / "knowledge_clusters.parquet"
            if knowledge_parquet.exists():
                size_mb = knowledge_parquet.stat().st_size / (1024 ** 2)
                storage_info["databases"]["knowledge"] = {
                    "path": str(knowledge_parquet),
                    "size_mb": round(size_mb, 2),
                    "exists": True,
                }
            
            # Check settings database
            settings_db = cache_path / "settings" / "settings.db"
            if settings_db.exists():
                size_mb = settings_db.stat().st_size / (1024 ** 2)
                storage_info["databases"]["settings"] = {
                    "path": str(settings_db),
                    "size_mb": round(size_mb, 2),
                    "exists": True,
                }
            
            # Calculate total cache size
            total_size = 0
            if cache_path.exists():
                for file in cache_path.rglob('*'):
                    if file.is_file():
                        total_size += file.stat().st_size
            
            storage_info["total_cache_size_mb"] = round(total_size / (1024 ** 2), 2)
            
            return storage_info
        
        except Exception as e:
            return {
                "work_path": "",
                "cache_path": "",
                "databases": {},
                "error": str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status
        
        Returns:
            Health status for all components
        """
        metrics = self.get_system_metrics()
        
        # Calculate health score
        health_score = 100
        issues = []
        
        # CPU check
        cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
        if cpu_usage > 90:
            health_score -= 30
            issues.append("High CPU usage")
        elif cpu_usage > 75:
            health_score -= 15
            issues.append("Elevated CPU usage")
        
        # Memory check
        memory_usage = metrics.get('memory', {}).get('usage_percent', 0)
        if memory_usage > 90:
            health_score -= 30
            issues.append("High memory usage")
        elif memory_usage > 80:
            health_score -= 15
            issues.append("Elevated memory usage")
        
        # Disk check
        disk_usage = metrics.get('disk', {}).get('usage_percent', 0)
        if disk_usage > 95:
            health_score -= 40
            issues.append("Critical disk usage")
        elif disk_usage > 85:
            health_score -= 20
            issues.append("High disk usage")
        
        # Determine overall status
        if health_score >= 90:
            status = "excellent"
            status_color = "green"
        elif health_score >= 70:
            status = "good"
            status_color = "blue"
        elif health_score >= 50:
            status = "warning"
            status_color = "yellow"
        else:
            status = "critical"
            status_color = "red"
        
        # Check service availability
        services = {
            "api": {
                "status": "running",
                "healthy": True,
            },
            "history_storage": {
                "status": "connected" if self.history_storage else "unavailable",
                "healthy": bool(self.history_storage),
            },
            "knowledge_manager": {
                "status": "connected" if self.knowledge_manager else "unavailable",
                "healthy": bool(self.knowledge_manager),
            },
        }
        
        return {
            "overall_status": status,
            "status_color": status_color,
            "health_score": max(0, health_score),
            "issues": issues,
            "services": services,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring overview
        
        Returns:
            Complete monitoring data
        """
        return {
            "system": self.get_system_metrics(),
            "chat": self.get_chat_activity(hours=24),
            "knowledge": self.get_knowledge_activity(),
            "storage": self.get_storage_info(),
            "health": self.get_health_status(),
            "timestamp": datetime.now().isoformat(),
        }


# Global instance
_monitor_tracker = None

def get_monitor_tracker() -> MonitorTracker:
    """Get or create global monitor tracker instance"""
    global _monitor_tracker
    if _monitor_tracker is None:
        _monitor_tracker = MonitorTracker()
    return _monitor_tracker

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
import threading

from sirchmunk.storage.knowledge_manager import KnowledgeManager
from api.components.history_storage import HistoryStorage
from sirchmunk.utils.constants import DEFAULT_WORK_PATH


class LLMUsageTracker:
    """
    Global tracker for LLM token usage and call statistics.
    Thread-safe singleton for tracking across the application.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize tracking data"""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.calls_by_model = {}
        self.last_call_time = None
        self.session_start = datetime.now()
        self._data_lock = threading.Lock()
    
    def record_usage(self, model: str, usage: Dict[str, int]):
        """
        Record token usage from an LLM call
        
        Args:
            model: Model name
            usage: Dictionary with prompt_tokens, completion_tokens, total_tokens
        """
        with self._data_lock:
            self.total_calls += 1
            self.last_call_time = datetime.now()
            
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            total = usage.get('total_tokens', input_tokens + output_tokens)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_tokens += total
            
            # Track by model
            if model not in self.calls_by_model:
                self.calls_by_model[model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            self.calls_by_model[model]["calls"] += 1
            self.calls_by_model[model]["input_tokens"] += input_tokens
            self.calls_by_model[model]["output_tokens"] += output_tokens
            self.calls_by_model[model]["total_tokens"] += total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        with self._data_lock:
            uptime_seconds = (datetime.now() - self.session_start).total_seconds()
            calls_per_minute = (self.total_calls / uptime_seconds * 60) if uptime_seconds > 0 else 0
            
            return {
                "total_calls": self.total_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "calls_per_minute": round(calls_per_minute, 2),
                "last_call_time": self.last_call_time.isoformat() if self.last_call_time else None,
                "session_start": self.session_start.isoformat(),
                "session_duration_minutes": round(uptime_seconds / 60, 1),
                "models": self.calls_by_model.copy(),
            }
    
    def reset(self):
        """Reset all statistics"""
        self._initialize()


# Global LLM usage tracker instance
llm_usage_tracker = LLMUsageTracker()


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

                if isinstance(session_time, datetime):
                    session_time = session_time.timestamp()
                elif isinstance(session_time, str):
                    try:
                        session_time = datetime.fromisoformat(session_time.replace('Z', '+00:00')).timestamp()
                    except (ValueError, TypeError):
                        session_time = 0

                # Count messages using message_count field (messages are not included in get_all_sessions)
                message_count = session.get('message_count', 0)
                total_messages += message_count

                # Check if recent
                if session_time >= threshold_ts:
                    recent_sessions.append({
                        "session_id": session.get('session_id'),
                        "title": session.get('title', 'Untitled'),
                        "message_count": message_count,
                        "created_at": session.get('created_at'),
                        "updated_at": session_time,  # Store as timestamp
                    })
                    active_count += 1

            # Sort by update time
            recent_sessions.sort(key=lambda x: x['updated_at'] if isinstance(x['updated_at'], (int, float)) else 0, reverse=True)

            # Get LLM usage stats
            llm_stats = llm_usage_tracker.get_stats()

            return {
                "total_sessions": len(all_sessions),
                "total_messages": total_messages,
                "recent_sessions": recent_sessions[:10],  # Top 10 most recent
                "active_sessions": active_count,
                "time_window_hours": hours,
                "llm_usage": llm_stats,
            }
        
        except Exception as e:
            print(f"[ERROR] Error getting chat activity in monitor: {e}")

            return {
                "total_sessions": 0,
                "total_messages": 0,
                "recent_sessions": [],
                "active_sessions": 0,
                "llm_usage": llm_usage_tracker.get_stats(),
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
            work_path = Path(os.getenv("WORK_PATH", DEFAULT_WORK_PATH))
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

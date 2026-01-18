# Copyright (c) ModelScope Contributors. All rights reserved.
"""
History API endpoints integrated with persistent storage
Provides unified history tracking with DuckDB backend
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta

# Import chat sessions and history storage from chat module
from .chat import chat_sessions, history_storage

router = APIRouter(prefix="/api/v1", tags=["history"])

# Create a second router for dashboard endpoints
from fastapi import APIRouter as FastAPIRouter
dashboard_router = FastAPIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/chat/sessions")
async def get_chat_sessions(limit: int = 20, offset: int = 0):
    """Get list of chat sessions from persistent storage"""
    # Get sessions from persistent storage
    sessions_list = history_storage.get_all_sessions(limit=limit, offset=offset)
    
    # Format for response
    formatted_sessions = []
    for session in sessions_list:
        # Get full session data to access messages for title generation
        full_session = history_storage.get_session(session["session_id"])
        
        title = session.get("title", "Chat Session")
        if full_session and full_session.get("messages"):
            first_user_message = next((m for m in full_session["messages"] if m["role"] == "user"), None)
            if first_user_message:
                title = first_user_message["content"][:50] + "..." if len(first_user_message["content"]) > 50 else first_user_message["content"]
        
        last_message = ""
        if full_session and full_session.get("messages"):
            last_msg = full_session["messages"][-1]
            last_message = last_msg["content"][:100] + "..." if len(last_msg["content"]) > 100 else last_msg["content"]
        
        # Convert ISO timestamps to Unix timestamps
        created_at = session["created_at"]
        updated_at = session["updated_at"]
        if isinstance(created_at, str):
            created_at = int(datetime.fromisoformat(created_at).timestamp())
        if isinstance(updated_at, str):
            updated_at = int(datetime.fromisoformat(updated_at).timestamp())
        
        formatted_sessions.append({
            "session_id": session["session_id"],
            "title": title,
            "message_count": session.get("message_count", 0),
            "last_message": last_message,
            "created_at": created_at,
            "updated_at": updated_at,
            "topics": [] # Placeholder
        })
    
    # Get total count
    total_count = history_storage.get_session_count()
    
    return {
        "success": True,
        "data": formatted_sessions,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total_count
        }
    }


@router.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get specific chat session details from persistent storage"""
    # Try to get from persistent storage first
    session = history_storage.get_session(session_id)
    
    # Fallback to in-memory cache if not in persistent storage
    if not session and session_id in chat_sessions:
        session = chat_sessions[session_id]
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Convert ISO timestamps to Unix timestamps for frontend compatibility
    messages_with_unix_timestamps = []
    for msg in session.get("messages", []):
        msg_copy = msg.copy()
        if "timestamp" in msg_copy:
            if isinstance(msg_copy["timestamp"], str):
                msg_copy["timestamp"] = int(datetime.fromisoformat(msg_copy["timestamp"]).timestamp())
        messages_with_unix_timestamps.append(msg_copy)
    
    # Handle created_at and updated_at
    created_at = session.get("created_at")
    updated_at = session.get("updated_at")
    
    if isinstance(created_at, str):
        created_at = int(datetime.fromisoformat(created_at).timestamp())
    if isinstance(updated_at, str):
        updated_at = int(datetime.fromisoformat(updated_at).timestamp())
    
    return {
        "success": True,
        "data": {
            "session_id": session["session_id"],
            "title": session.get("title", "Chat Session"),
            "messages": messages_with_unix_timestamps,
            "settings": session.get("settings", {}),
            "created_at": created_at,
            "updated_at": updated_at
        }
    }


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a specific chat session from both memory and persistent storage"""
    # Delete from persistent storage
    success = history_storage.delete_session(session_id)
    
    # Also delete from in-memory cache
    if session_id in chat_sessions:
        chat_sessions.pop(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return {
        "success": True,
        "message": "Chat session deleted successfully",
        "data": {
            "session_id": session_id
        }
    }


@router.get("/history/search")
async def search_history(query: str, limit: int = 20):
    """Search chat history by content"""
    sessions = history_storage.search_sessions(query, limit=limit)
    
    # Format for response
    formatted_sessions = []
    for session in sessions:
        # Convert timestamps
        created_at = session["created_at"]
        updated_at = session["updated_at"]
        if isinstance(created_at, str):
            created_at = int(datetime.fromisoformat(created_at).timestamp())
        if isinstance(updated_at, str):
            updated_at = int(datetime.fromisoformat(updated_at).timestamp())
        
        formatted_sessions.append({
            "session_id": session["session_id"],
            "title": session.get("title", "Chat Session"),
            "message_count": session.get("message_count", 0),
            "created_at": created_at,
            "updated_at": updated_at,
        })
    
    return {
        "success": True,
        "data": formatted_sessions,
        "query": query
    }


@router.get("/history/stats")
async def get_history_statistics():
    """Get history statistics from persistent storage"""
    total_sessions = history_storage.get_session_count()
    
    # Get recent sessions (last 7 days)
    all_sessions = history_storage.get_all_sessions(limit=1000)
    week_ago = datetime.now() - timedelta(days=7)
    
    recent_sessions = []
    total_messages = 0
    
    for session in all_sessions:
        updated_at = session.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        if updated_at > week_ago:
            recent_sessions.append(session)
        
        total_messages += session.get("message_count", 0)
    
    return {
        "success": True,
        "data": {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "recent_activity": {
                "last_7_days": len(recent_sessions),
                "daily_average": len(recent_sessions) / 7
            }
        }
    }


@dashboard_router.get("/recent")
async def get_recent_activity(limit: int = 50, type: Optional[str] = None):
    """
    Get recent activity (chat sessions)
    
    Query params:
        limit: Maximum number of items to return
        type: Filter by type (currently only "chat" is supported)
    """
    try:
        # Get recent sessions
        sessions_list = history_storage.get_all_sessions(limit=limit, offset=0)
        
        # Format as activity items
        activities = []
        for session in sessions_list:
            # Get full session data
            full_session = history_storage.get_session(session["session_id"])
            
            title = session.get("title", "Chat Session")
            if full_session and full_session.get("messages"):
                first_user_message = next((m for m in full_session["messages"] if m["role"] == "user"), None)
                if first_user_message:
                    title = first_user_message["content"][:50] + "..." if len(first_user_message["content"]) > 50 else first_user_message["content"]
            
            # Convert timestamps
            created_at = session["created_at"]
            if isinstance(created_at, str):
                created_at = int(datetime.fromisoformat(created_at).timestamp())
            
            activities.append({
                "id": session["session_id"],
                "type": "chat",
                "title": title,
                "timestamp": created_at,
                "message_count": session.get("message_count", 0),
            })
    
        return {
            "success": True,
                "data": activities,
                "count": len(activities),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
Mock API endpoints for history functionality
Provides unified history tracking across all system activities
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/api/v1", tags=["history"])

# Mock activity history data
activity_history = []

# Generate mock history entries
def generate_mock_history():
    """Generate mock history entries for different activity types"""
    global activity_history
    
    if activity_history:  # Already generated
        return
    
    base_time = datetime.now()
    
    # Generate solve activities
    for i in range(15):
        timestamp = base_time - timedelta(hours=random.randint(1, 72))
        activity_history.append({
            "id": f"solve_{i:03d}",
            "type": "solve",
            "title": f"Solved: {random.choice(['Machine Learning Basics', 'Python Data Structures', 'Algorithm Optimization', 'Neural Network Architecture', 'Statistical Analysis'])}",
            "summary": f"Comprehensive solution provided using {random.choice(['ai_textbook', 'python_docs', 'research_papers'])} knowledge base",
            "timestamp": int(timestamp.timestamp()),
            "content": {
                "question": f"Sample question about {random.choice(['ML', 'Python', 'algorithms'])}",
                "kb_name": random.choice(["ai_textbook", "python_docs", "research_papers"]),
                "tokens_used": random.randint(500, 2000),
                "processing_time": random.uniform(2, 15)
            }
        })
    
    # Generate question activities
    for i in range(12):
        timestamp = base_time - timedelta(hours=random.randint(1, 96))
        activity_history.append({
            "id": f"question_{i:03d}",
            "type": "question",
            "title": f"Generated {random.randint(5, 20)} Questions",
            "summary": f"Created {random.choice(['multiple choice', 'short answer', 'essay'])} questions on {random.choice(['AI fundamentals', 'Python programming', 'data science'])}",
            "timestamp": int(timestamp.timestamp()),
            "content": {
                "question_count": random.randint(5, 20),
                "question_types": random.sample(["multiple_choice", "short_answer", "essay"], 2),
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "kb_names": random.sample(["ai_textbook", "python_docs", "research_papers"], 2)
            }
        })
    
    # Generate research activities
    for i in range(8):
        timestamp = base_time - timedelta(hours=random.randint(1, 120))
        activity_history.append({
            "id": f"research_{i:03d}",
            "type": "research",
            "title": f"Research: {random.choice(['Artificial Intelligence Trends', 'Climate Change Impact', 'Quantum Computing Applications', 'Sustainable Technology'])}",
            "summary": f"Comprehensive research using {random.randint(3, 5)} tools, analyzed {random.randint(20, 60)} sources",
            "timestamp": int(timestamp.timestamp()),
            "content": {
                "topic": f"Research topic {i+1}",
                "mode": random.choice(["quick", "comprehensive", "deep"]),
                "tools_used": random.sample(["web_search", "academic_search", "knowledge_base", "expert_interview"], 3),
                "sources_analyzed": random.randint(20, 60),
                "processing_time": random.uniform(15, 45)
            }
        })
    
    # Generate chat activities
    for i in range(10):
        timestamp = base_time - timedelta(hours=random.randint(1, 48))
        activity_history.append({
            "id": f"chat_{i:03d}",
            "type": "chat",
            "title": f"Chat Session {i+1}",
            "summary": f"Interactive conversation with {random.randint(5, 25)} messages",
            "timestamp": int(timestamp.timestamp()),
            "content": {
                "message_count": random.randint(5, 25),
                "session_duration": random.randint(300, 1800),  # 5-30 minutes
                "topics_discussed": random.sample(["AI", "Programming", "Research", "Learning"], 2)
            }
        })
    
    # Sort by timestamp (newest first)
    activity_history.sort(key=lambda x: x["timestamp"], reverse=True)

# Mock chat sessions
chat_sessions = []

def generate_mock_chat_sessions():
    """Generate mock chat sessions"""
    global chat_sessions
    
    if chat_sessions:  # Already generated
        return
    
    base_time = datetime.now()
    
    for i in range(8):
        created_time = base_time - timedelta(hours=random.randint(1, 168))  # Last week
        updated_time = created_time + timedelta(minutes=random.randint(5, 120))
        
        chat_sessions.append({
            "session_id": f"session_{i:03d}",
            "title": f"Chat Session {i+1}",
            "message_count": random.randint(3, 30),
            "last_message": random.choice([
                "Thank you for the explanation!",
                "Can you provide more details about this topic?",
                "That makes sense now.",
                "What are the practical applications?",
                "How does this relate to other concepts?"
            ]),
            "created_at": int(created_time.timestamp()),
            "updated_at": int(updated_time.timestamp()),
            "topics": random.sample(["Machine Learning", "Python", "Data Science", "AI Ethics"], 2)
        })
    
    # Sort by updated_at (most recent first)
    chat_sessions.sort(key=lambda x: x["updated_at"], reverse=True)

@router.get("/dashboard/recent")
async def get_recent_activities(
    limit: int = 20,
    offset: int = 0,
    type: Optional[str] = None
):
    """Get recent activity history"""
    generate_mock_history()
    
    # Filter by type if specified
    filtered_activities = activity_history
    if type and type != "all":
        filtered_activities = [a for a in activity_history if a["type"] == type]
    
    # Apply pagination
    paginated_activities = filtered_activities[offset:offset + limit]
    
    return {
        "success": True,
        "data": paginated_activities,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(filtered_activities)
        }
    }

@router.get("/history/activities")
async def get_activity_history(
    limit: int = 50,
    offset: int = 0,
    type: Optional[str] = None,
    search: Optional[str] = None
):
    """Get comprehensive activity history with search and filtering"""
    generate_mock_history()
    
    filtered_activities = activity_history
    
    # Filter by type
    if type and type != "all":
        filtered_activities = [a for a in filtered_activities if a["type"] == type]
    
    # Filter by search query
    if search:
        search_lower = search.lower()
        filtered_activities = [
            a for a in filtered_activities
            if search_lower in a["title"].lower() or 
               search_lower in a["summary"].lower()
        ]
    
    # Apply pagination
    paginated_activities = filtered_activities[offset:offset + limit]
    
    return {
        "success": True,
        "data": paginated_activities,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(filtered_activities)
        },
        "filters": {
            "type": type,
            "search": search
        }
    }

@router.get("/history/activity/{activity_id}")
async def get_activity_detail(activity_id: str):
    """Get detailed information for a specific activity"""
    generate_mock_history()
    
    activity = next((a for a in activity_history if a["id"] == activity_id), None)
    if not activity:
        raise HTTPException(status_code=404, detail="Activity not found")
    
    # Add additional details based on activity type
    detailed_activity = activity.copy()
    
    if activity["type"] == "solve":
        detailed_activity["content"]["solution"] = f"""
# Solution for: {activity['content']['question']}

This is a comprehensive solution that addresses all aspects of your question.

## Analysis
The problem requires understanding of fundamental concepts and their practical applications.

## Implementation
Here's a step-by-step approach to solve this problem:

1. **Data Preparation**: Ensure your data is clean and properly formatted
2. **Algorithm Selection**: Choose the most appropriate method
3. **Implementation**: Apply the selected approach
4. **Validation**: Test and verify the results

## Code Example
```python
def solve_problem(data):
    # Process the input data
    processed_data = preprocess(data)
    
    # Apply the algorithm
    result = algorithm(processed_data)
    
    return result
```

## Conclusion
This solution provides a robust approach that can be adapted to similar problems.
"""
    
    elif activity["type"] == "question":
        detailed_activity["content"]["sample_questions"] = [
            {
                "id": f"q_{i}",
                "type": random.choice(["multiple_choice", "short_answer", "essay"]),
                "question": f"Sample question {i+1} about the topic",
                "difficulty": random.choice(["easy", "medium", "hard"])
            }
            for i in range(3)
        ]
    
    elif activity["type"] == "research":
        detailed_activity["content"]["report_sections"] = [
            "Executive Summary",
            "Current State Analysis", 
            "Key Findings",
            "Challenges and Opportunities",
            "Future Outlook",
            "Recommendations"
        ]
    
    return {
        "success": True,
        "data": detailed_activity
    }

@router.get("/chat/sessions")
async def get_chat_sessions(limit: int = 20, offset: int = 0):
    """Get chat session history"""
    generate_mock_chat_sessions()
    
    # Apply pagination
    paginated_sessions = chat_sessions[offset:offset + limit]
    
    return {
        "success": True,
        "data": paginated_sessions,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(chat_sessions)
        }
    }

@router.get("/chat/session/{session_id}")
async def get_chat_session_detail(session_id: str):
    """Get detailed chat session information"""
    generate_mock_chat_sessions()
    
    session = next((s for s in chat_sessions if s["session_id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Generate mock messages for the session
    messages = []
    message_count = session["message_count"]
    
    for i in range(message_count):
        is_user = i % 2 == 0
        timestamp = session["created_at"] + (i * 60)  # 1 minute apart
        
        if is_user:
            content = random.choice([
                "Can you explain machine learning concepts?",
                "How do neural networks work?",
                "What are the best practices for data preprocessing?",
                "Can you help me understand this algorithm?",
                "What's the difference between supervised and unsupervised learning?"
            ])
        else:
            content = random.choice([
                "I'd be happy to explain machine learning concepts! Machine learning is...",
                "Neural networks are computational models inspired by biological neural networks...",
                "Great question! Data preprocessing is crucial for machine learning success...",
                "Certainly! Let me break down this algorithm step by step...",
                "The key difference between supervised and unsupervised learning is..."
            ])
        
        messages.append({
            "id": f"msg_{i:03d}",
            "type": "user" if is_user else "assistant",
            "content": content,
            "timestamp": timestamp,
            "tokens": random.randint(10, 200) if not is_user else None
        })
    
    detailed_session = session.copy()
    detailed_session["messages"] = messages
    detailed_session["total_tokens"] = sum(m.get("tokens", 0) for m in messages)
    
    return {
        "success": True,
        "data": detailed_session
    }

@router.post("/chat/session/{session_id}/load")
async def load_chat_session(session_id: str):
    """Load chat session for continuation"""
    generate_mock_chat_sessions()
    
    session = next((s for s in chat_sessions if s["session_id"] == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # In a real implementation, this would restore the session state
    # For mock purposes, we'll just return success
    
    return {
        "success": True,
        "message": f"Chat session '{session['title']}' loaded successfully",
        "data": {
            "session_id": session_id,
            "title": session["title"],
            "message_count": session["message_count"],
            "loaded_at": datetime.now().isoformat()
        }
    }

@router.get("/history/stats")
async def get_history_statistics():
    """Get history statistics and analytics"""
    generate_mock_history()
    generate_mock_chat_sessions()
    
    # Calculate statistics
    total_activities = len(activity_history)
    total_chat_sessions = len(chat_sessions)
    
    # Count by type
    type_counts = {}
    for activity in activity_history:
        activity_type = activity["type"]
        type_counts[activity_type] = type_counts.get(activity_type, 0) + 1
    
    # Recent activity (last 7 days)
    week_ago = datetime.now() - timedelta(days=7)
    recent_activities = [
        a for a in activity_history 
        if a["timestamp"] > week_ago.timestamp()
    ]
    
    # Most active day
    daily_counts = {}
    for activity in recent_activities:
        date_key = datetime.fromtimestamp(activity["timestamp"]).strftime("%Y-%m-%d")
        daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
    
    most_active_day = max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None
    
    return {
        "success": True,
        "data": {
            "total_activities": total_activities,
            "total_chat_sessions": total_chat_sessions,
            "type_distribution": type_counts,
            "recent_activity": {
                "last_7_days": len(recent_activities),
                "daily_average": len(recent_activities) / 7,
                "most_active_day": {
                    "date": most_active_day[0] if most_active_day else None,
                    "count": most_active_day[1] if most_active_day else 0
                }
            },
            "usage_patterns": {
                "peak_hours": [14, 15, 16, 20, 21],  # 2-4 PM, 8-9 PM
                "preferred_types": sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            }
        }
    }

@router.delete("/history/activity/{activity_id}")
async def delete_activity(activity_id: str):
    """Delete specific activity from history"""
    global activity_history
    
    activity_index = next((i for i, a in enumerate(activity_history) if a["id"] == activity_id), None)
    if activity_index is None:
        raise HTTPException(status_code=404, detail="Activity not found")
    
    deleted_activity = activity_history.pop(activity_index)
    
    return {
        "success": True,
        "message": "Activity deleted successfully",
        "data": deleted_activity
    }

@router.post("/history/clear")
async def clear_history(request: Dict[str, Any]):
    """Clear history based on criteria"""
    global activity_history, chat_sessions
    
    clear_type = request.get("type", "all")  # all, activities, chat
    older_than_days = request.get("older_than_days")
    
    deleted_count = 0
    
    if clear_type in ["all", "activities"]:
        if older_than_days:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            original_count = len(activity_history)
            activity_history = [a for a in activity_history if a["timestamp"] > cutoff_timestamp]
            deleted_count += original_count - len(activity_history)
        else:
            deleted_count += len(activity_history)
            activity_history = []
    
    if clear_type in ["all", "chat"]:
        if older_than_days:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            original_count = len(chat_sessions)
            chat_sessions = [s for s in chat_sessions if s["updated_at"] > cutoff_timestamp]
            deleted_count += original_count - len(chat_sessions)
        else:
            deleted_count += len(chat_sessions)
            chat_sessions = []
    
    return {
        "success": True,
        "message": f"Cleared {deleted_count} items from history",
        "data": {
            "deleted_count": deleted_count,
            "clear_type": clear_type,
            "older_than_days": older_than_days
        }
    }

@router.get("/history/export")
async def export_history(
    format: str = "json",  # json, csv, markdown
    type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Export history data in various formats"""
    generate_mock_history()
    generate_mock_chat_sessions()
    
    # Filter data based on parameters
    filtered_data = activity_history
    
    if type and type != "all":
        filtered_data = [a for a in filtered_data if a["type"] == type]
    
    if start_date:
        start_timestamp = datetime.fromisoformat(start_date).timestamp()
        filtered_data = [a for a in filtered_data if a["timestamp"] >= start_timestamp]
    
    if end_date:
        end_timestamp = datetime.fromisoformat(end_date).timestamp()
        filtered_data = [a for a in filtered_data if a["timestamp"] <= end_timestamp]
    
    # Generate export content based on format
    if format == "json":
        export_content = json.dumps({
            "activities": filtered_data,
            "chat_sessions": chat_sessions,
            "exported_at": datetime.now().isoformat(),
            "total_items": len(filtered_data) + len(chat_sessions)
        }, indent=2)
        filename = f"deeptutor_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    elif format == "csv":
        # Simple CSV format for activities
        csv_lines = ["ID,Type,Title,Summary,Timestamp"]
        for activity in filtered_data:
            csv_lines.append(f"{activity['id']},{activity['type']},{activity['title']},{activity['summary']},{activity['timestamp']}")
        
        export_content = "\n".join(csv_lines)
        filename = f"deeptutor_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    elif format == "markdown":
        md_lines = [
            "# DeepTutor Activity History",
            f"\nExported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal Activities: {len(filtered_data)}",
            "\n---\n"
        ]
        
        for activity in filtered_data:
            md_lines.extend([
                f"## {activity['title']}",
                f"**Type:** {activity['type'].title()}",
                f"**Date:** {datetime.fromtimestamp(activity['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Summary:** {activity['summary']}",
                "\n---\n"
            ])
        
        export_content = "\n".join(md_lines)
        filename = f"deeptutor_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")
    
    return {
        "success": True,
        "data": {
            "filename": filename,
            "format": format,
            "content": export_content,
            "size": len(export_content.encode('utf-8')),
            "items_exported": len(filtered_data),
            "download_url": f"/api/v1/history/download/{uuid.uuid4().hex}"
        }
    }
"""
Unified API endpoints for chat and search functionality
Provides WebSocket endpoint for real-time chat conversations with integrated search
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import json
import asyncio
import uuid
from datetime import datetime
import random
import os
import threading
from pathlib import Path

# Try to import tkinter for file dialogs
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Import search functionality
try:
    from sirchmunk.search import AgenticSearch
    SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Failed to import sirchmunk.search: {e}")
    raise

router = APIRouter(prefix="/api/v1", tags=["chat", "search"])

# Mock chat sessions storage
chat_sessions = {}

# Active WebSocket connections
class ChatConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

# Unified log callback management
class LogCallbackManager:
    """Centralized management for all log callback functions"""

    @staticmethod
    async def create_search_log_callback(websocket: WebSocket, manager: ChatConnectionManager):
        """Create search log callback for chat WebSocket"""
        async def search_log_callback(level: str, log_message: str):
            await manager.send_personal_message(json.dumps({
                "type": "search_log",
                "level": level,
                "message": log_message,
                "timestamp": datetime.now().isoformat()
            }), websocket)
            await asyncio.sleep(0.1)  # Small delay for proper streaming
        return search_log_callback

    @staticmethod
    async def create_websocket_log_callback(websocket: WebSocket):
        """Create log callback for search WebSocket"""
        async def log_callback(level: str, message: str):
            await websocket.send_text(json.dumps({
                "type": "log",
                "level": level,
                "message": message,
                "timestamp": asyncio.get_event_loop().time()
            }))
            # Small delay to ensure proper streaming
            await asyncio.sleep(0.1)
        return log_callback

    @staticmethod
    async def create_rest_log_callback():
        """Create log callback for REST API (silent operation)"""
        async def rest_log_callback(level: str, message: str):
            # For REST API, we use silent operation
            # This ensures consistency with other search instances
            pass
        return rest_log_callback

manager = ChatConnectionManager()

# Search-related models and functions
class SearchRequest(BaseModel):
    query: str
    search_paths: Union[str, List[str]]  # Expects absolute file/directory paths from user's local filesystem
    mode: Optional[str] = "DEEP"
    max_depth: Optional[int] = 5
    top_k_files: Optional[int] = 3

def get_search_instance(log_callback=None):
    """Get configured search instance with optional log callback"""
    if log_callback:
        return AgenticSearch(log_callback=log_callback)
    return AgenticSearch()

def open_file_dialog(dialog_type: str = "files", multiple: bool = True) -> List[str]:
    """
    Open native file dialog using tkinter
    Returns list of absolute file paths from user's local filesystem
    """
    if not TKINTER_AVAILABLE:
        return []
    
    selected_paths = []
    
    def run_dialog():
        nonlocal selected_paths
        try:
            # Create root window but hide it
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            if dialog_type == "files":
                if multiple:
                    # Multiple file selection
                    files = filedialog.askopenfilenames(
                        title="Select Files",
                        filetypes=[
                            ("All files", "*.*"),
                            ("Text files", "*.txt"),
                            ("Python files", "*.py"),
                            ("JSON files", "*.json"),
                            ("CSV files", "*.csv"),
                            ("PDF files", "*.pdf"),
                            ("Word documents", "*.docx"),
                            ("Excel files", "*.xlsx")
                        ]
                    )
                    selected_paths = list(files) if files else []
                else:
                    # Single file selection
                    file_path = filedialog.askopenfilename(
                        title="Select File",
                        filetypes=[
                            ("All files", "*.*"),
                            ("Text files", "*.txt"),
                            ("Python files", "*.py"),
                            ("JSON files", "*.json"),
                            ("CSV files", "*.csv"),
                            ("PDF files", "*.pdf"),
                            ("Word documents", "*.docx"),
                            ("Excel files", "*.xlsx")
                        ]
                    )
                    selected_paths = [file_path] if file_path else []
            
            elif dialog_type == "directory":
                # Directory selection
                dir_path = filedialog.askdirectory(
                    title="Select Directory"
                )
                selected_paths = [dir_path] if dir_path else []
            
            root.destroy()
            
        except Exception as e:
            print(f"Error in file dialog: {e}")
            selected_paths = []
    
    # Run dialog in main thread
    if threading.current_thread() is threading.main_thread():
        run_dialog()
    else:
        # If not in main thread, we need to handle this differently
        # For now, return empty list as tkinter requires main thread
        return []
    
    return selected_paths

def generate_mock_response(message: str, enable_rag: bool = False, enable_web_search: bool = False, kb_name: str = "") -> str:
    """Generate mock chat response based on message content"""
    message_lower = message.lower()
    
    # Context-aware responses based on keywords
    if any(keyword in message_lower for keyword in ["hello", "hi", "hey", "greetings"]):
        return "Hello! I'm Sirchmunk, your AI learning assistant. How can I help you today?"
    
    elif any(keyword in message_lower for keyword in ["machine learning", "ml", "ai", "artificial intelligence"]):
        return """# Machine Learning Overview

Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.

## Key Concepts

### Types of Learning
- **Supervised Learning**: Learning with labeled examples
- **Unsupervised Learning**: Finding patterns in unlabeled data
- **Reinforcement Learning**: Learning through trial and error with rewards

### Popular Algorithms
- Linear Regression for prediction
- Decision Trees for classification
- Neural Networks for complex patterns
- K-Means for clustering

## Mathematical Foundation

The core optimization problem in ML:

$$\\min_{\\theta} \\frac{1}{m} \\sum_{i=1}^{m} L(h_{\\theta}(x^{(i)}), y^{(i)})$$

Where:
- $\\theta$ represents model parameters
- $L$ is the loss function
- $h_{\\theta}$ is the hypothesis function

## Practical Applications
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis

Would you like me to dive deeper into any specific aspect of machine learning?"""

    elif any(keyword in message_lower for keyword in ["python", "programming", "code"]):
        return """# Python Programming Guide

Python is an excellent language for beginners and experts alike, known for its readability and versatility.

## Basic Syntax

```python
# Variables and data types
name = "Alice"
age = 25
scores = [85, 92, 78, 96]

# Control structures
if age >= 18:
    print("Adult")
else:
    print("Minor")

# Functions
def greet(name):
    return f"Hello, {name}!"

# Classes
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def study(self, subject):
        return f"{self.name} is studying {subject}"
```

## Popular Libraries
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning
- **Django/Flask**: Web development

## Best Practices
1. Follow PEP 8 style guidelines
2. Write clear, descriptive variable names
3. Use docstrings for documentation
4. Handle exceptions properly
5. Write unit tests

What specific Python topic would you like to explore further?"""

    elif any(keyword in message_lower for keyword in ["help", "assist", "support"]):
        return """# How I Can Help You

I'm Sirchmunk, your AI learning companion! Here's what I can do:

## 🎓 Learning Support
- Explain complex concepts in simple terms
- Provide step-by-step solutions
- Generate practice questions
- Create study guides

## 💻 Programming Help
- Debug code issues
- Explain algorithms
- Suggest best practices
- Review code structure

## 📊 Research Assistance
- Summarize academic papers
- Find relevant resources
- Generate research ideas
- Create comprehensive reports

## 🧮 Problem Solving
- Mathematical problem solving
- Logical reasoning
- Critical thinking exercises
- Real-world applications

## 🔧 Available Tools
- **Smart Solver**: Multi-step problem solving
- **Question Generator**: Create practice questions
- **Research Assistant**: Deep research reports
- **Co-Writer**: Collaborative writing
- **Guided Learning**: Structured tutorials

What would you like to work on today?"""

    else:
        # Generic helpful response
        return f"""Thank you for your question about "{message}". 

I understand you're looking for information on this topic. Let me provide a comprehensive response:

## Analysis

Your question touches on important concepts that are worth exploring in detail. Based on the context, I can help you understand the key principles and practical applications.

## Key Points

1. **Fundamental Concepts**: Understanding the basic principles is crucial for building a solid foundation.

2. **Practical Applications**: Real-world examples help connect theory to practice.

3. **Best Practices**: Following established guidelines ensures optimal results.

4. **Common Challenges**: Being aware of potential pitfalls helps avoid mistakes.

## Next Steps

To dive deeper into this topic, I recommend:
- Exploring related concepts
- Practicing with examples
- Asking specific follow-up questions
- Applying the knowledge to real projects

Would you like me to elaborate on any particular aspect, or do you have more specific questions about this topic?"""

# WebSocket endpoint for chat with integrated search
@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat conversations with integrated search"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            message = request_data.get("message", "")
            session_id = request_data.get("session_id")
            history = request_data.get("history", [])
            kb_name = request_data.get("kb_name", "")
            enable_rag = request_data.get("enable_rag", False)
            enable_web_search = request_data.get("enable_web_search", False)
            
            # Generate or use existing session ID
            if not session_id:
                session_id = f"chat_{uuid.uuid4().hex[:8]}"
            
            # Send session ID to client
            await manager.send_personal_message(json.dumps({
                "type": "session",
                "session_id": session_id
            }), websocket)
            
            # Store session data
            if session_id not in chat_sessions:
                chat_sessions[session_id] = {
                    "session_id": session_id,
                    "title": f"Chat Session",
                    "messages": [],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "settings": {
                        "kb_name": kb_name,
                        "enable_rag": enable_rag,
                        "enable_web_search": enable_web_search
                    }
                }
            
            # Update session with new message
            session = chat_sessions[session_id]
            session["messages"].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            session["updated_at"] = datetime.now().isoformat()
            
            # Handle real search if enabled and search available
            # if enable_rag and kb_name and SEARCH_AVAILABLE:
            if SEARCH_AVAILABLE:
                try:
                    # Create log callback for streaming search logs
                    search_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)

                    # Send search start signal
                    await manager.send_personal_message(json.dumps({
                        "type": "status",
                        "stage": "search",
                        "message": f"🔍 Starting real-time search in: {kb_name}"
                    }), websocket)

                    # Create search instance with log callback
                    search_engine = get_search_instance(log_callback=search_log_callback)
                    
                    # Perform real search with streaming logs
                    # Parse comma-separated paths from kb_name or use default paths
                    if kb_name and isinstance(kb_name, str) and "," in kb_name:
                        search_paths = [path.strip() for path in kb_name.split(",")]
                    elif kb_name and isinstance(kb_name, str):
                        search_paths = [kb_name]
                    else:
                        # Default search paths if no kb_name provided
                        search_paths = ["/home/project/src", "/home/project/web"]
                        await search_log_callback("info", "🔧 Using default search paths: src and web directories")

                    # Add debug log
                    await search_log_callback("info", f"🔧 Debug: Parsed search paths: {search_paths}")

                    # Debug log before executing search
                    print(f"[DEBUG] About to execute search_engine.search() with query: {message}")
                    print(f"[DEBUG] Search paths: {search_paths}")

                    search_result = await search_engine.search(
                        query=message,
                        search_paths=search_paths,
                        max_depth=5,
                        top_k_files=3,
                        verbose=True
                    )
                    
                    # Send search completion
                    await manager.send_personal_message(json.dumps({
                        "type": "search_complete",
                        "message": "✅ Search completed successfully"
                    }), websocket)
                    
                    # Use search result as response
                    response = search_result
                    
                except Exception as e:
                    # Send search error
                    await manager.send_personal_message(json.dumps({
                        "type": "search_error",
                        "message": f"❌ Search failed: {str(e)}"
                    }), websocket)
                    
                    # Fallback to mock response
                    response = generate_mock_response(message, enable_rag, enable_web_search, kb_name)
            else:
                # Simulate processing stages for mock responses
                stages = []
                
                if enable_rag:
                    stages.append({"stage": "rag", "message": f"Searching knowledge base: {kb_name}"})
                
                if enable_web_search:
                    stages.append({"stage": "web", "message": "Searching the web for relevant information"})
                
                stages.append({"stage": "generating", "message": "Generating response"})
                
                # Send status updates
                for stage_info in stages:
                    await manager.send_personal_message(json.dumps({
                        "type": "status",
                        **stage_info
                    }), websocket)
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                
                # Generate mock response
                response = generate_mock_response(message, enable_rag, enable_web_search, kb_name)
            
            # Simulate streaming response
            words = response.split()
            streamed_content = ""
            
            for i, word in enumerate(words):
                streamed_content += word + " "
                await manager.send_personal_message(json.dumps({
                    "type": "stream",
                    "content": word + " "
                }), websocket)
                
                # Add small delay for realistic streaming
                if i % 3 == 0:  # Every 3 words
                    await asyncio.sleep(0.1)
            
            # Generate mock sources if RAG or web search enabled
            sources = {}
            if enable_rag and kb_name:
                sources["rag"] = [
                    {
                        "kb_name": kb_name,
                        "content": f"Relevant content from {kb_name} knowledge base"
                    }
                ]
            
            if enable_web_search:
                sources["web"] = [
                    {
                        "url": "https://example.com/article1",
                        "title": "Comprehensive Guide to the Topic",
                        "snippet": "This article provides detailed information about the subject matter..."
                    },
                    {
                        "url": "https://example.com/article2", 
                        "title": "Advanced Concepts and Applications",
                        "snippet": "Exploring advanced techniques and real-world applications..."
                    }
                ]
            
            # Send sources if available
            if sources:
                await manager.send_personal_message(json.dumps({
                    "type": "sources",
                    **sources
                }), websocket)
            
            # Send final result
            await manager.send_personal_message(json.dumps({
                "type": "result",
                "content": response.strip(),
                "session_id": session_id
            }), websocket)
            
            # Store assistant response in session
            session["messages"].append({
                "role": "assistant",
                "content": response.strip(),
                "sources": sources if sources else None,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        try:
            await manager.send_personal_message(json.dumps({
                "type": "error",
                "message": f"An error occurred: {str(e)}"
            }), websocket)
        except:
            pass
        manager.disconnect(websocket)

# REST API endpoints for search functionality
@router.post("/search")
async def search_files(request: SearchRequest):
    """Search files using Sirchmunk search engine"""
    try:
        # Create log callback using LogCallbackManager
        rest_log_callback = await LogCallbackManager.create_rest_log_callback()

        # Get search instance with log callback
        search_engine = get_search_instance(log_callback=rest_log_callback)

        # Convert search_paths to appropriate format
        if isinstance(request.search_paths, str):
            search_paths = request.search_paths
        else:
            search_paths = request.search_paths

        # Perform search
        result = await search_engine.search(
            query=request.query,
            search_paths=search_paths,
            max_depth=request.max_depth,
            top_k_files=request.top_k_files
        )

        return {
            "success": True,
            "data": {
                "query": request.query,
                "search_paths": request.search_paths,
                "result": result
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.websocket("/search/ws")
async def search_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time search with streaming logs"""
    await websocket.accept()
    
    try:
        while True:
            # Receive search request from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            query = request_data.get("query", "")
            search_paths = request_data.get("search_paths", [])
            max_depth = request_data.get("max_depth", 5)
            top_k_files = request_data.get("top_k_files", 3)
            
            if not query or not search_paths:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Query and search_paths are required"
                }))
                continue
            
            # Create log callback for streaming
            log_callback = await LogCallbackManager.create_websocket_log_callback(websocket)

            try:
                # Create search instance with log callback
                search_engine = get_search_instance(log_callback=log_callback)
                
                # Send start signal
                await websocket.send_text(json.dumps({
                    "type": "start",
                    "query": query,
                    "search_paths": search_paths
                }))
                
                # Perform search with streaming logs
                result = await search_engine.search(
                    query=query,
                    search_paths=search_paths,
                    max_depth=max_depth,
                    top_k_files=top_k_files,
                    verbose=True
                )
                
                # Send final result
                await websocket.send_text(json.dumps({
                    "type": "result",
                    "success": True,
                    "data": {
                        "query": query,
                        "search_paths": search_paths,
                        "result": result
                    }
                }))
                
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Search failed: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"WebSocket error: {str(e)}"
            }))
        except:
            pass

# File picker endpoints
@router.post("/file-picker")
async def open_file_picker(request: Dict[str, Any]):
    """
    Open native file picker dialog using tkinter
    Returns real absolute paths from user's local filesystem
    """
    if not TKINTER_AVAILABLE:
        return {
            "success": False,
            "error": "Tkinter not available on this system",
            "data": []
        }
    
    dialog_type = request.get("type", "files")  # "files" or "directory"
    multiple = request.get("multiple", True)
    
    try:
        # Get absolute paths from user's local filesystem
        selected_paths = open_file_dialog(dialog_type, multiple)
        
        # Convert to absolute paths and validate they exist
        validated_paths = []
        for path in selected_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                validated_paths.append(abs_path)
        
        return {
            "success": True,
            "data": {
                "paths": validated_paths,
                "count": len(validated_paths),
                "type": dialog_type,
                "multiple": multiple
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to open file picker: {str(e)}",
            "data": []
        }

@router.get("/file-picker/status")
async def get_file_picker_status():
    """Check if file picker is available on this system"""
    return {
        "success": True,
        "data": {
            "tkinter_available": TKINTER_AVAILABLE,
            "supported_types": ["files", "directory"] if TKINTER_AVAILABLE else [],
            "features": {
                "multiple_files": TKINTER_AVAILABLE,
                "directory_selection": TKINTER_AVAILABLE,
                "absolute_paths": TKINTER_AVAILABLE
            }
        }
    }

# Chat session management endpoints
@router.get("/chat/sessions")
async def get_chat_sessions(limit: int = 20, offset: int = 0):
    """Get list of chat sessions"""
    sessions_list = list(chat_sessions.values())
    # Sort by updated_at (most recent first)
    sessions_list.sort(key=lambda x: x["updated_at"], reverse=True)
    
    # Apply pagination
    paginated_sessions = sessions_list[offset:offset + limit]
    
    # Format for response
    formatted_sessions = []
    for session in paginated_sessions:
        last_message = ""
        if session["messages"]:
            last_msg = session["messages"][-1]
            last_message = last_msg["content"][:100] + "..." if len(last_msg["content"]) > 100 else last_msg["content"]
        
        formatted_sessions.append({
            "session_id": session["session_id"],
            "title": session.get("title", "Chat Session"),
            "message_count": len(session["messages"]),
            "last_message": last_message,
            "created_at": int(datetime.fromisoformat(session["created_at"]).timestamp()),
            "updated_at": int(datetime.fromisoformat(session["updated_at"]).timestamp()),
            "topics": ["AI", "Learning"]  # Mock topics
        })
    
    return {
        "success": True,
        "data": formatted_sessions,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(sessions_list)
        }
    }

@router.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get specific chat session details"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    session = chat_sessions[session_id]
    
    return {
        "success": True,
        "data": {
            "session_id": session["session_id"],
            "title": session.get("title", "Chat Session"),
            "messages": session["messages"],
            "settings": session.get("settings", {}),
            "created_at": session["created_at"],
            "updated_at": session["updated_at"]
        }
    }

@router.post("/chat/sessions/{session_id}/load")
async def load_chat_session(session_id: str):
    """Load chat session for continuation"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    session = chat_sessions[session_id]
    
    return {
        "success": True,
        "message": f"Chat session loaded successfully",
        "data": {
            "session_id": session_id,
            "title": session.get("title", "Chat Session"),
            "message_count": len(session["messages"]),
            "loaded_at": datetime.now().isoformat()
        }
    }

# Legacy search endpoints for backward compatibility
@router.get("/search/{kb_name}/suggestions")
async def get_search_suggestions(kb_name: str, query: str, limit: int = 8):
    """Get search suggestions - kept for backward compatibility"""
    # For now, return empty suggestions since we're using real file search
    if not query or len(query.strip()) < 2:
        return {
            "success": True,
            "data": [],
            "query": query
        }

    return {
        "success": True,
        "data": [],
        "query": query,
        "total_matches": 0
    }

@router.get("/search/knowledge-bases")
async def get_knowledge_bases():
    """Get list of available knowledge bases for search"""
    # Return empty list since we're using direct file paths now
    return {
        "success": True,
        "data": []
    }
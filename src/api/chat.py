"""
Mock API endpoints for chat functionality
Provides WebSocket endpoint for real-time chat conversations
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Any, List
import json
import asyncio
import uuid
from datetime import datetime
import random

router = APIRouter(prefix="/api/v1", tags=["chat"])

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

manager = ChatConnectionManager()

def generate_mock_response(message: str, enable_rag: bool = False, enable_web_search: bool = False, kb_name: str = "") -> str:
    """Generate mock chat response based on message content"""
    message_lower = message.lower()
    
    # Context-aware responses based on keywords
    if any(keyword in message_lower for keyword in ["hello", "hi", "hey", "greetings"]):
        return "Hello! I'm OpenCowork, your AI learning assistant. How can I help you today?"
    
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

    elif any(keyword in message_lower for keyword in ["math", "mathematics", "calculus", "algebra"]):
        return """# Mathematics in Computer Science

Mathematics forms the foundation of computer science and AI. Let me explain some key areas:

## Linear Algebra
Essential for machine learning and graphics:

$$\\mathbf{A}\\mathbf{x} = \\mathbf{b}$$

Where $\\mathbf{A}$ is a matrix, $\\mathbf{x}$ is a vector of unknowns.

## Calculus
Used in optimization and neural networks:

$$\\frac{\\partial}{\\partial w} J(w) = 0$$

Finding the minimum of cost function $J(w)$.

## Statistics & Probability
Core to data science and ML:

$$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$

Bayes' theorem for conditional probability.

## Discrete Mathematics
Important for algorithms and logic:
- Graph theory for networks
- Combinatorics for counting
- Logic for reasoning systems

Which mathematical area interests you most?"""

    elif any(keyword in message_lower for keyword in ["help", "assist", "support"]):
        return """# How I Can Help You

I'm OpenCowork, your AI learning companion! Here's what I can do:

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

@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat conversations"""
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
            
            # Simulate processing stages
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
            
            # Generate response
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
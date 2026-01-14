"""
Mock API endpoints for knowledge base management
Provides CRUD operations for knowledge bases and document management
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime
import asyncio
import random

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

# Mock data storage
knowledge_bases = [
    {
        "id": "kb_001",
        "name": "ai_textbook",
        "display_name": "AI Textbook",
        "description": "Comprehensive AI and machine learning textbook",
        "document_count": 156,
        "size": "2.3GB",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-10T12:30:00Z",
        "is_default": True,
        "status": "ready",
        "progress": 100,
        "statistics": {
            "raw_documents": 156,
            "images": 23,
            "content_lists": 89,
            "rag_initialized": True,
            "rag": {
                "chunks": 2340,
                "entities": 567,
                "relations": 234
            }
        }
    },
    {
        "id": "kb_002",
        "name": "python_docs",
        "display_name": "Python Documentation",
        "description": "Official Python documentation and tutorials",
        "document_count": 89,
        "size": "1.2GB",
        "created_at": "2024-01-05T10:15:00Z",
        "updated_at": "2024-01-12T08:45:00Z",
        "is_default": False,
        "status": "ready",
        "progress": 100,
        "statistics": {
            "raw_documents": 89,
            "images": 12,
            "content_lists": 45,
            "rag_initialized": True,
            "rag": {
                "chunks": 1890,
                "entities": 423,
                "relations": 178
            }
        }
    },
    {
        "id": "kb_003",
        "name": "research_papers",
        "display_name": "Research Papers",
        "description": "Collection of academic research papers",
        "document_count": 234,
        "size": "3.8GB",
        "created_at": "2024-01-08T14:20:00Z",
        "updated_at": "2024-01-13T16:10:00Z",
        "is_default": False,
        "status": "processing",
        "progress": 75,
        "statistics": {
            "raw_documents": 234,
            "images": 45,
            "content_lists": 123,
            "rag_initialized": False,
            "rag": {
                "chunks": 1750,
                "entities": 298,
                "relations": 145
            }
        }
    }
]

processing_status = {}

@router.get("/list")
async def list_knowledge_bases():
    """Get list of all knowledge bases"""
    return {
        "success": True,
        "data": knowledge_bases,
        "total": len(knowledge_bases)
    }

@router.get("/{kb_name}")
async def get_knowledge_base(kb_name: str):
    """Get specific knowledge base details"""
    kb = next((kb for kb in knowledge_bases if kb["name"] == kb_name), None)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return {
        "success": True,
        "data": kb
    }

@router.post("/create")
async def create_knowledge_base(request: Dict[str, Any]):
    """Create new knowledge base"""
    name = request.get("name")
    display_name = request.get("display_name", name)
    description = request.get("description", "")
    
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    
    # Check if name already exists
    if any(kb["name"] == name for kb in knowledge_bases):
        raise HTTPException(status_code=400, detail="Knowledge base name already exists")
    
    new_kb = {
        "id": f"kb_{uuid.uuid4().hex[:6]}",
        "name": name,
        "display_name": display_name,
        "description": description,
        "document_count": 0,
        "size": "0MB",
        "created_at": datetime.now().isoformat() + "Z",
        "updated_at": datetime.now().isoformat() + "Z",
        "is_default": False,
        "status": "empty",
        "progress": 0
    }
    
    knowledge_bases.append(new_kb)
    
    return {
        "success": True,
        "data": new_kb,
        "message": "Knowledge base created successfully"
    }

@router.delete("/{kb_name}")
async def delete_knowledge_base(kb_name: str):
    """Delete knowledge base"""
    global knowledge_bases
    
    kb_index = next((i for i, kb in enumerate(knowledge_bases) if kb["name"] == kb_name), None)
    if kb_index is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Don't allow deletion of default knowledge base
    if knowledge_bases[kb_index]["is_default"]:
        raise HTTPException(status_code=400, detail="Cannot delete default knowledge base")
    
    deleted_kb = knowledge_bases.pop(kb_index)
    
    return {
        "success": True,
        "message": f"Knowledge base '{kb_name}' deleted successfully",
        "data": deleted_kb
    }

@router.post("/{kb_name}/upload")
async def upload_documents(kb_name: str, files: List[UploadFile] = File(...)):
    """Upload documents to knowledge base"""
    kb = next((kb for kb in knowledge_bases if kb["name"] == kb_name), None)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Simulate file processing
    upload_id = str(uuid.uuid4())
    processing_status[upload_id] = {
        "status": "processing",
        "progress": 0,
        "total_files": len(files),
        "processed_files": 0,
        "current_file": files[0].filename if files else None
    }
    
    # Update KB status
    kb["status"] = "processing"
    kb["progress"] = 0
    
    return {
        "success": True,
        "upload_id": upload_id,
        "message": f"Started processing {len(files)} files",
        "data": {
            "files": [{"name": f.filename, "size": f.size} for f in files]
        }
    }

@router.get("/{kb_name}/upload/{upload_id}/status")
async def get_upload_status(kb_name: str, upload_id: str):
    """Get upload processing status"""
    if upload_id not in processing_status:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    status = processing_status[upload_id]
    
    # Simulate progress update
    if status["status"] == "processing":
        status["progress"] = min(status["progress"] + 10, 100)
        status["processed_files"] = min(status["processed_files"] + 1, status["total_files"])
        
        if status["progress"] >= 100:
            status["status"] = "completed"
            # Update KB
            kb = next((kb for kb in knowledge_bases if kb["name"] == kb_name), None)
            if kb:
                kb["status"] = "ready"
                kb["progress"] = 100
                kb["document_count"] += status["total_files"]
                kb["updated_at"] = datetime.now().isoformat() + "Z"
    
    return {
        "success": True,
        "data": status
    }

@router.post("/{kb_name}/clear_progress")
async def clear_progress(kb_name: str):
    """Clear stuck processing progress"""
    kb = next((kb for kb in knowledge_bases if kb["name"] == kb_name), None)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    kb["status"] = "ready"
    kb["progress"] = 100
    
    return {
        "success": True,
        "message": "Progress cleared successfully"
    }

@router.get("/{kb_name}/documents")
async def list_documents(kb_name: str, page: int = 1, limit: int = 20):
    """List documents in knowledge base"""
    kb = next((kb for kb in knowledge_bases if kb["name"] == kb_name), None)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Mock document list
    documents = [
        {
            "id": f"doc_{i:03d}",
            "name": f"document_{i}.pdf",
            "size": f"{(i * 123) % 1000 + 100}KB",
            "type": "pdf",
            "uploaded_at": datetime.now().isoformat() + "Z",
            "status": "processed"
        }
        for i in range(1, min(kb["document_count"] + 1, 21))
    ]
    
    start = (page - 1) * limit
    end = start + limit
    paginated_docs = documents[start:end]
    
    return {
        "success": True,
        "data": paginated_docs,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": kb["document_count"],
            "pages": (kb["document_count"] + limit - 1) // limit
        }
    }


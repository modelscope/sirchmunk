"""
Hub API endpoints for Sirchmunk
Handles file management, synchronization, and hub operations
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/api/v1/hub", tags=["hub"])

# Mock hub files data
MOCK_HUB_FILES = [
    {
        "id": "hub_001",
        "name": "Project_Proposal_2024.pdf",
        "type": "pdf",
        "size": "2.4 MB",
        "modified": "2 hours ago",
        "status": "synced",
        "url": "https://example.com/files/project_proposal.pdf",
        "created_at": "2024-01-13T10:30:00Z",
        "updated_at": "2024-01-13T15:45:00Z"
    },
    {
        "id": "hub_002", 
        "name": "Meeting_Notes_Jan.docx",
        "type": "docx",
        "size": "856 KB",
        "modified": "1 day ago",
        "status": "pending",
        "url": "https://example.com/files/meeting_notes.docx",
        "created_at": "2024-01-12T14:20:00Z",
        "updated_at": "2024-01-12T16:30:00Z"
    },
    {
        "id": "hub_003",
        "name": "Q4_Analytics.xlsx",
        "type": "xlsx", 
        "size": "1.8 MB",
        "modified": "3 days ago",
        "status": "synced",
        "url": "https://example.com/files/q4_analytics.xlsx",
        "created_at": "2024-01-10T09:15:00Z",
        "updated_at": "2024-01-10T11:45:00Z"
    },
    {
        "id": "hub_004",
        "name": "Presentation_Draft.pptx",
        "type": "pptx",
        "size": "5.2 MB", 
        "modified": "5 days ago",
        "status": "error",
        "url": None,
        "created_at": "2024-01-08T16:00:00Z",
        "updated_at": "2024-01-08T17:30:00Z"
    },
    {
        "id": "hub_005",
        "name": "Research_Data.csv",
        "type": "csv",
        "size": "3.1 MB",
        "modified": "1 week ago", 
        "status": "synced",
        "url": "https://example.com/files/research_data.csv",
        "created_at": "2024-01-06T13:45:00Z",
        "updated_at": "2024-01-06T14:20:00Z"
    }
]

@router.get("/files")
async def list_hub_files(
    status: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List files in the hub
    
    Args:
        status: Filter by file status (synced, pending, error)
        file_type: Filter by file type (pdf, docx, xlsx, etc.)
        limit: Maximum number of files to return
        offset: Number of files to skip
    """
    try:
        files = MOCK_HUB_FILES.copy()
        
        # Apply filters
        if status:
            files = [f for f in files if f["status"] == status]
        
        if file_type:
            files = [f for f in files if f["type"].lower() == file_type.lower()]
        
        # Apply pagination
        total = len(files)
        files = files[offset:offset + limit]
        
        return {
            "success": True,
            "data": files,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list hub files: {str(e)}")

@router.get("/files/{file_id}")
async def get_hub_file(file_id: str):
    """Get detailed information about a specific hub file"""
    try:
        file = next((f for f in MOCK_HUB_FILES if f["id"] == file_id), None)
        
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Add additional metadata for detailed view
        detailed_file = {
            **file,
            "metadata": {
                "file_hash": f"sha256_{random.randint(100000, 999999)}",
                "version": "1.2.0",
                "tags": ["work", "project", "important"],
                "permissions": {
                    "read": True,
                    "write": True,
                    "share": True
                }
            },
            "sync_history": [
                {
                    "timestamp": "2024-01-13T15:45:00Z",
                    "action": "sync",
                    "status": "success"
                },
                {
                    "timestamp": "2024-01-13T10:30:00Z", 
                    "action": "upload",
                    "status": "success"
                }
            ]
        }
        
        return {
            "success": True,
            "data": detailed_file,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hub file: {str(e)}")

@router.post("/sync")
async def sync_hub_files(request: Optional[Dict[str, Any]] = None):
    """
    Synchronize files with the hub
    
    Args:
        request: Optional sync configuration
    """
    try:
        # Simulate sync process
        sync_results = {
            "sync_id": str(uuid.uuid4()),
            "started_at": datetime.now().isoformat(),
            "status": "completed",
            "summary": {
                "total_files": len(MOCK_HUB_FILES),
                "synced": len([f for f in MOCK_HUB_FILES if f["status"] == "synced"]),
                "pending": len([f for f in MOCK_HUB_FILES if f["status"] == "pending"]),
                "errors": len([f for f in MOCK_HUB_FILES if f["status"] == "error"]),
                "new_files": random.randint(0, 3),
                "updated_files": random.randint(1, 5)
            },
            "duration_ms": random.randint(1500, 3000),
            "completed_at": (datetime.now() + timedelta(seconds=2)).isoformat()
        }
        
        # Update some file statuses to simulate sync effects
        for file in MOCK_HUB_FILES:
            if file["status"] == "pending" and random.random() > 0.5:
                file["status"] = "synced"
                file["updated_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "data": sync_results,
            "message": "Hub synchronization completed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync hub files: {str(e)}")

@router.post("/upload")
async def upload_to_hub(request: Dict[str, Any]):
    """
    Upload a file to the hub
    
    Args:
        request: Upload request with file information
    """
    try:
        file_name = request.get("file_name", "untitled")
        file_type = request.get("file_type", "txt")
        file_size = request.get("file_size", "1 KB")
        
        new_file = {
            "id": f"hub_{str(uuid.uuid4())[:8]}",
            "name": file_name,
            "type": file_type,
            "size": file_size,
            "modified": "just now",
            "status": "pending",
            "url": f"https://example.com/files/{file_name.lower().replace(' ', '_')}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        MOCK_HUB_FILES.insert(0, new_file)
        
        return {
            "success": True,
            "data": new_file,
            "message": "File uploaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@router.delete("/files/{file_id}")
async def delete_hub_file(file_id: str):
    """Delete a file from the hub"""
    try:
        global MOCK_HUB_FILES
        
        file_index = next((i for i, f in enumerate(MOCK_HUB_FILES) if f["id"] == file_id), None)
        
        if file_index is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        deleted_file = MOCK_HUB_FILES.pop(file_index)
        
        return {
            "success": True,
            "data": {
                "deleted_file": deleted_file,
                "deletion_timestamp": datetime.now().isoformat()
            },
            "message": "File deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@router.get("/status")
async def get_hub_status():
    """Get overall hub status and statistics"""
    try:
        status_data = {
            "hub_status": "online",
            "last_sync": "2024-01-13T15:45:00Z",
            "storage": {
                "used": "45.2 GB",
                "total": "100 GB",
                "usage_percentage": 45.2
            },
            "file_statistics": {
                "total_files": len(MOCK_HUB_FILES),
                "by_status": {
                    "synced": len([f for f in MOCK_HUB_FILES if f["status"] == "synced"]),
                    "pending": len([f for f in MOCK_HUB_FILES if f["status"] == "pending"]),
                    "error": len([f for f in MOCK_HUB_FILES if f["status"] == "error"])
                },
                "by_type": {}
            },
            "sync_history": [
                {
                    "timestamp": "2024-01-13T15:45:00Z",
                    "status": "success",
                    "files_synced": 12
                },
                {
                    "timestamp": "2024-01-13T10:30:00Z",
                    "status": "success", 
                    "files_synced": 8
                },
                {
                    "timestamp": "2024-01-12T18:20:00Z",
                    "status": "partial",
                    "files_synced": 5
                }
            ]
        }
        
        # Calculate file type statistics
        type_counts = {}
        for file in MOCK_HUB_FILES:
            file_type = file["type"]
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        status_data["file_statistics"]["by_type"] = type_counts
        
        return {
            "success": True,
            "data": status_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hub status: {str(e)}")
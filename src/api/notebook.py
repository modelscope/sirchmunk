"""
Mock API endpoints for notebook management functionality
Provides CRUD operations for notebooks and records management
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime
import random

router = APIRouter(prefix="/api/v1/notebook", tags=["notebook"])

# Mock notebook data
notebooks = [
    {
        "id": "nb_001",
        "name": "AI Research Notes",
        "description": "Collection of AI and machine learning research notes",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-13T10:30:00Z",
        "record_count": 25,
        "tags": ["AI", "Machine Learning", "Research"]
    },
    {
        "id": "nb_002", 
        "name": "Python Programming",
        "description": "Python coding examples and best practices",
        "created_at": "2024-01-05T14:20:00Z",
        "updated_at": "2024-01-12T16:45:00Z",
        "record_count": 18,
        "tags": ["Python", "Programming", "Code Examples"]
    },
    {
        "id": "nb_003",
        "name": "Project Ideas",
        "description": "Innovative project concepts and implementations",
        "created_at": "2024-01-08T09:15:00Z",
        "updated_at": "2024-01-13T08:20:00Z",
        "record_count": 12,
        "tags": ["Ideas", "Projects", "Innovation"]
    }
]

# Mock notebook records
notebook_records = {
    "nb_001": [
        {
            "id": "rec_001",
            "notebook_id": "nb_001",
            "title": "Deep Learning Fundamentals",
            "content": "# Deep Learning Fundamentals\n\nDeep learning is a subset of machine learning...",
            "type": "research_result",
            "source": "Research Session",
            "created_at": "2024-01-10T14:30:00Z",
            "updated_at": "2024-01-10T14:30:00Z",
            "tags": ["Deep Learning", "Neural Networks"]
        },
        {
            "id": "rec_002",
            "notebook_id": "nb_001",
            "title": "Transformer Architecture Analysis",
            "content": "# Transformer Architecture\n\nThe transformer model revolutionized NLP...",
            "type": "solver_result",
            "source": "Problem Solver",
            "created_at": "2024-01-11T09:15:00Z",
            "updated_at": "2024-01-11T09:15:00Z",
            "tags": ["Transformers", "NLP", "Architecture"]
        }
    ],
    "nb_002": [
        {
            "id": "rec_003",
            "notebook_id": "nb_002",
            "title": "Python Best Practices",
            "content": "# Python Best Practices\n\n## Code Style\n- Follow PEP 8...",
            "type": "question_result",
            "source": "Question Generator",
            "created_at": "2024-01-09T11:20:00Z",
            "updated_at": "2024-01-09T11:20:00Z",
            "tags": ["Python", "Best Practices", "PEP 8"]
        }
    ],
    "nb_003": [
        {
            "id": "rec_004",
            "notebook_id": "nb_003",
            "title": "AI-Powered Education Platform",
            "content": "# AI-Powered Education Platform\n\nConcept for personalized learning...",
            "type": "idea_result",
            "source": "Idea Generator",
            "created_at": "2024-01-12T16:45:00Z",
            "updated_at": "2024-01-12T16:45:00Z",
            "tags": ["Education", "AI", "Platform"]
        }
    ]
}

@router.get("/list")
async def list_notebooks():
    """Get list of all notebooks"""
    # Add summary statistics
    notebooks_with_stats = []
    for notebook in notebooks:
        records = notebook_records.get(notebook["id"], [])
        notebook_copy = notebook.copy()
        notebook_copy["record_count"] = len(records)
        
        # Calculate type distribution
        type_counts = {}
        for record in records:
            record_type = record.get("type", "unknown")
            type_counts[record_type] = type_counts.get(record_type, 0) + 1
        
        notebook_copy["type_distribution"] = type_counts
        notebooks_with_stats.append(notebook_copy)
    
    return {
        "success": True,
        "data": notebooks_with_stats,
        "total": len(notebooks)
    }

@router.get("/{notebook_id}")
async def get_notebook_detail(notebook_id: str):
    """Get detailed notebook information with records"""
    notebook = next((nb for nb in notebooks if nb["id"] == notebook_id), None)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    records = notebook_records.get(notebook_id, [])
    
    return {
        "success": True,
        "data": {
            "notebook": notebook,
            "records": records,
            "statistics": {
                "total_records": len(records),
                "types": list(set(r.get("type", "unknown") for r in records)),
                "latest_update": max([r["updated_at"] for r in records]) if records else notebook["updated_at"]
            }
        }
    }

@router.post("/create")
async def create_notebook(request: Dict[str, Any]):
    """Create new notebook"""
    name = request.get("name")
    description = request.get("description", "")
    tags = request.get("tags", [])
    
    if not name:
        raise HTTPException(status_code=400, detail="Notebook name is required")
    
    # Check if name already exists
    if any(nb["name"] == name for nb in notebooks):
        raise HTTPException(status_code=400, detail="Notebook name already exists")
    
    new_notebook = {
        "id": f"nb_{uuid.uuid4().hex[:6]}",
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat() + "Z",
        "updated_at": datetime.now().isoformat() + "Z",
        "record_count": 0,
        "tags": tags
    }
    
    notebooks.append(new_notebook)
    notebook_records[new_notebook["id"]] = []
    
    return {
        "success": True,
        "data": new_notebook,
        "message": "Notebook created successfully"
    }

@router.put("/{notebook_id}")
async def update_notebook(notebook_id: str, request: Dict[str, Any]):
    """Update notebook information"""
    notebook = next((nb for nb in notebooks if nb["id"] == notebook_id), None)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    # Update fields
    if "name" in request:
        # Check for name conflicts
        if any(nb["name"] == request["name"] and nb["id"] != notebook_id for nb in notebooks):
            raise HTTPException(status_code=400, detail="Notebook name already exists")
        notebook["name"] = request["name"]
    
    if "description" in request:
        notebook["description"] = request["description"]
    
    if "tags" in request:
        notebook["tags"] = request["tags"]
    
    notebook["updated_at"] = datetime.now().isoformat() + "Z"
    
    return {
        "success": True,
        "data": notebook,
        "message": "Notebook updated successfully"
    }

@router.delete("/{notebook_id}")
async def delete_notebook(notebook_id: str):
    """Delete notebook and all its records"""
    global notebooks
    
    notebook_index = next((i for i, nb in enumerate(notebooks) if nb["id"] == notebook_id), None)
    if notebook_index is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    deleted_notebook = notebooks.pop(notebook_index)
    
    # Delete associated records
    if notebook_id in notebook_records:
        del notebook_records[notebook_id]
    
    return {
        "success": True,
        "message": f"Notebook '{deleted_notebook['name']}' deleted successfully",
        "data": deleted_notebook
    }

@router.post("/{notebook_id}/records")
async def add_record_to_notebook(notebook_id: str, request: Dict[str, Any]):
    """Add new record to notebook"""
    notebook = next((nb for nb in notebooks if nb["id"] == notebook_id), None)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    title = request.get("title")
    content = request.get("content")
    record_type = request.get("type", "manual")
    source = request.get("source", "Manual Entry")
    tags = request.get("tags", [])
    
    if not title or not content:
        raise HTTPException(status_code=400, detail="Title and content are required")
    
    new_record = {
        "id": f"rec_{uuid.uuid4().hex[:6]}",
        "notebook_id": notebook_id,
        "title": title,
        "content": content,
        "type": record_type,
        "source": source,
        "created_at": datetime.now().isoformat() + "Z",
        "updated_at": datetime.now().isoformat() + "Z",
        "tags": tags
    }
    
    if notebook_id not in notebook_records:
        notebook_records[notebook_id] = []
    
    notebook_records[notebook_id].append(new_record)
    
    # Update notebook
    notebook["updated_at"] = datetime.now().isoformat() + "Z"
    notebook["record_count"] = len(notebook_records[notebook_id])
    
    return {
        "success": True,
        "data": new_record,
        "message": "Record added successfully"
    }

@router.delete("/{notebook_id}/records/{record_id}")
async def delete_record(notebook_id: str, record_id: str):
    """Delete specific record from notebook"""
    if notebook_id not in notebook_records:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    records = notebook_records[notebook_id]
    record_index = next((i for i, r in enumerate(records) if r["id"] == record_id), None)
    
    if record_index is None:
        raise HTTPException(status_code=404, detail="Record not found")
    
    deleted_record = records.pop(record_index)
    
    # Update notebook record count
    notebook = next((nb for nb in notebooks if nb["id"] == notebook_id), None)
    if notebook:
        notebook["record_count"] = len(records)
        notebook["updated_at"] = datetime.now().isoformat() + "Z"
    
    return {
        "success": True,
        "message": "Record deleted successfully",
        "data": deleted_record
    }

@router.post("/import")
async def import_records(request: Dict[str, Any]):
    """Import records from one notebook to another"""
    source_notebook_id = request.get("source_notebook_id")
    target_notebook_id = request.get("target_notebook_id")
    record_ids = request.get("record_ids", [])
    
    if not source_notebook_id or not target_notebook_id:
        raise HTTPException(status_code=400, detail="Source and target notebook IDs are required")
    
    # Validate notebooks exist
    source_notebook = next((nb for nb in notebooks if nb["id"] == source_notebook_id), None)
    target_notebook = next((nb for nb in notebooks if nb["id"] == target_notebook_id), None)
    
    if not source_notebook:
        raise HTTPException(status_code=404, detail="Source notebook not found")
    if not target_notebook:
        raise HTTPException(status_code=404, detail="Target notebook not found")
    
    # Get source records
    source_records = notebook_records.get(source_notebook_id, [])
    records_to_import = [r for r in source_records if r["id"] in record_ids]
    
    if not records_to_import:
        raise HTTPException(status_code=400, detail="No valid records found to import")
    
    # Import records (create copies with new IDs)
    imported_records = []
    for record in records_to_import:
        new_record = record.copy()
        new_record["id"] = f"rec_{uuid.uuid4().hex[:6]}"
        new_record["notebook_id"] = target_notebook_id
        new_record["created_at"] = datetime.now().isoformat() + "Z"
        new_record["updated_at"] = datetime.now().isoformat() + "Z"
        
        if target_notebook_id not in notebook_records:
            notebook_records[target_notebook_id] = []
        
        notebook_records[target_notebook_id].append(new_record)
        imported_records.append(new_record)
    
    # Update target notebook
    target_notebook["record_count"] = len(notebook_records[target_notebook_id])
    target_notebook["updated_at"] = datetime.now().isoformat() + "Z"
    
    return {
        "success": True,
        "data": imported_records,
        "message": f"Successfully imported {len(imported_records)} records"
    }

@router.get("/{notebook_id}/export/markdown")
async def export_notebook_markdown(notebook_id: str):
    """Export notebook as markdown"""
    notebook = next((nb for nb in notebooks if nb["id"] == notebook_id), None)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    records = notebook_records.get(notebook_id, [])
    
    # Generate markdown content
    markdown_content = f"# {notebook['name']}\n\n"
    if notebook['description']:
        markdown_content += f"{notebook['description']}\n\n"
    
    markdown_content += f"**Created:** {notebook['created_at']}\n"
    markdown_content += f"**Updated:** {notebook['updated_at']}\n"
    markdown_content += f"**Records:** {len(records)}\n\n"
    
    if notebook.get('tags'):
        markdown_content += f"**Tags:** {', '.join(notebook['tags'])}\n\n"
    
    markdown_content += "---\n\n"
    
    for record in records:
        markdown_content += f"## {record['title']}\n\n"
        markdown_content += f"**Type:** {record['type']}\n"
        markdown_content += f"**Source:** {record['source']}\n"
        markdown_content += f"**Created:** {record['created_at']}\n\n"
        
        if record.get('tags'):
            markdown_content += f"**Tags:** {', '.join(record['tags'])}\n\n"
        
        markdown_content += f"{record['content']}\n\n"
        markdown_content += "---\n\n"
    
    return {
        "success": True,
        "data": {
            "filename": f"{notebook['name'].replace(' ', '_')}.md",
            "content": markdown_content,
            "size": len(markdown_content.encode('utf-8'))
        }
    }

@router.get("/search")
async def search_notebooks(
    query: str = "",
    notebook_id: Optional[str] = None,
    record_type: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """Search notebooks and records"""
    results = []
    
    # Search in specific notebook or all notebooks
    notebooks_to_search = [notebook_id] if notebook_id else list(notebook_records.keys())
    
    for nb_id in notebooks_to_search:
        notebook = next((nb for nb in notebooks if nb["id"] == nb_id), None)
        if not notebook:
            continue
            
        records = notebook_records.get(nb_id, [])
        
        for record in records:
            # Apply filters
            if record_type and record.get("type") != record_type:
                continue
                
            if tags:
                tag_list = [t.strip() for t in tags.split(",")]
                if not any(tag in record.get("tags", []) for tag in tag_list):
                    continue
            
            # Search in title and content
            if query:
                query_lower = query.lower()
                if (query_lower not in record.get("title", "").lower() and 
                    query_lower not in record.get("content", "").lower()):
                    continue
            
            # Add to results
            result = record.copy()
            result["notebook_name"] = notebook["name"]
            results.append(result)
    
    # Sort by relevance (updated_at for now)
    results.sort(key=lambda x: x["updated_at"], reverse=True)
    
    # Paginate
    paginated_results = results[offset:offset + limit]
    
    return {
        "success": True,
        "data": paginated_results,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(results)
        },
        "query_info": {
            "query": query,
            "notebook_id": notebook_id,
            "record_type": record_type,
            "tags": tags
        }
    }
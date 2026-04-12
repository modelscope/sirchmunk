"""File upload and collection management API endpoints."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .file_service import (
    CollectionInfo,
    DiskUsageInfo,
    FileMetadata,
    FileStorageService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/files", tags=["files"])

# Module-level singleton (matches knowledge.py pattern)
_service: Optional[FileStorageService] = None


def _get_service() -> FileStorageService:
    """Lazy-init the FileStorageService singleton."""
    global _service
    if _service is None:
        _service = FileStorageService()
    return _service


def _check_enabled() -> None:
    """Raise 403 if upload feature is disabled."""
    if not FileStorageService.is_enabled():
        raise HTTPException(
            status_code=403,
            detail="File upload is disabled. Set SIRCHMUNK_UPLOAD_ENABLED=true to enable.",
        )


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(..., description="One or more files to upload"),
    collection: str = Form(default="default", description="Collection name (alphanumeric, dashes, underscores)"),
    paths: List[str] = Form(default=[], description="Relative paths preserving directory structure"),
):
    """Batch upload files to a named collection."""
    _check_enabled()
    svc = _get_service()

    try:
        svc.validate_collection_name(collection)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    results: List[dict] = []
    errors: List[dict] = []

    for i, upload_file in enumerate(files):
        try:
            content = await upload_file.read()
            relative_path = paths[i] if i < len(paths) else None
            meta = await svc.save_file(
                collection,
                upload_file.filename or "unnamed",
                content,
                relative_path=relative_path,
            )
            results.append(meta.model_dump())
        except ValueError as e:
            errors.append({"name": upload_file.filename, "error": str(e)})
        except Exception as e:
            logger.exception("Failed to save file %s", upload_file.filename)
            errors.append({"name": upload_file.filename, "error": str(e)})

    col_path = str(svc.get_collection_path(collection)) if results else None

    return {
        "success": len(results) > 0,
        "data": {
            "collection": collection,
            "collection_path": col_path,
            "uploaded": results,
            "errors": errors,
            "total_uploaded": len(results),
            "total_errors": len(errors),
        },
    }


@router.get("/collections")
async def list_collections(include_files: bool = False):
    """List all upload collections."""
    _check_enabled()
    svc = _get_service()
    try:
        collections = svc.list_collections()
        data = [c.model_dump() for c in collections]
        if include_files:
            for item in data:
                try:
                    files = svc.list_files(item["name"])
                    item["files"] = [f.model_dump() for f in files]
                except FileNotFoundError:
                    item["files"] = []
        return {
            "success": True,
            "count": len(data),
            "data": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{name}")
async def get_collection(name: str):
    """List files in a specific collection."""
    _check_enabled()
    svc = _get_service()
    try:
        files = svc.list_files(name)
        return {
            "success": True,
            "data": {
                "collection": name,
                "path": str(svc.get_collection_path(name)),
                "files": [f.model_dump() for f in files],
                "count": len(files),
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{name}/path")
async def get_collection_path(name: str):
    """Get the server-side filesystem path for a collection (for use in search API paths parameter)."""
    _check_enabled()
    svc = _get_service()
    try:
        path = svc.get_collection_path(name)
        return {
            "success": True,
            "data": {"collection": name, "path": str(path)},
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete an entire collection and all its files."""
    _check_enabled()
    svc = _get_service()
    try:
        count = svc.delete_collection(name)
        return {
            "success": True,
            "message": f"Collection '{name}' deleted ({count} files removed)",
            "data": {"collection": name, "files_deleted": count},
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete a single file by its ID."""
    _check_enabled()
    svc = _get_service()
    try:
        collection, filename = svc.delete_file(file_id)
        return {
            "success": True,
            "message": f"File '{filename}' deleted from collection '{collection}'",
            "data": {"file_id": file_id, "collection": collection, "name": filename},
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/usage")
async def get_usage():
    """Get disk usage statistics for upload storage."""
    _check_enabled()
    svc = _get_service()
    try:
        usage = svc.get_disk_usage()
        return {
            "success": True,
            "data": usage.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

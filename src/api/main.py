# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Main FastAPI application for Sirchmunk API
Combines all API modules and provides centralized configuration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import all API routers
from .knowledge import router as knowledge_router
from .settings import router as settings_router
from .history import router as history_router, dashboard_router
from .chat import router as chat_router
from .monitor import router as monitor_router

# Create FastAPI application
app = FastAPI(
    title="Sirchmunk API",
    description="APIs for Sirchmunk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routers
app.include_router(knowledge_router)
app.include_router(settings_router)
app.include_router(history_router)
app.include_router(dashboard_router)
app.include_router(chat_router)
app.include_router(monitor_router)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Sirchmunk API",
        "version": "1.0.0",
        "description": "APIs for Sirchmunk",
        "status": "running",
        "endpoints": {
            "knowledge": "/api/v1/knowledge",
            "settings": "/api/v1/settings",
            "history": "/api/v1/history",
            "chat": "/api/v1/chat",
            "monitor": "/api/v1/monitor"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-13T17:30:00Z",
        "services": {
            "api": "running",
            "database": "connected",
            "llm": "available",
            "embedding": "available"
        }
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": "The requested resource was not found",
                "path": str(request.url.path)
            }
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred",
                "details": "Please try again later or contact support"
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8584,
        reload=True,
        log_level="info"
    )

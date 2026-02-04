#!/usr/bin/env python3
"""
FastAPI server runner for Sirchmunk API
Provides a simple way to start the API server with proper configuration
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for running the FastAPI server"""
    try:
        import uvicorn
        from sirchmunk.api.main import app
        
        # Get configuration from environment
        host = os.environ.get("API_HOST", "0.0.0.0")
        port = int(os.environ.get("API_PORT", os.environ.get("BACKEND_PORT", "8584")))
        reload = os.environ.get("API_RELOAD", "true").lower() == "true"
        log_level = os.environ.get("API_LOG_LEVEL", "info")
        
        print(f"🚀 Starting Sirchmunk API server...")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Reload: {reload}")
        print(f"   Log Level: {log_level}")
        print(f"   Docs: http://{host}:{port}/docs")
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        print("   Please install required dependencies:")
        print("   pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
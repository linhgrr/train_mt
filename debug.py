#!/usr/bin/env python3
"""
Simple test script to debug import issues
"""
import os
import sys

print("Python version:", sys.version)
print("PORT environment variable:", os.environ.get("PORT", "Not set"))

try:
    import uvicorn
    print("✓ uvicorn imported successfully")
except ImportError as e:
    print("✗ Failed to import uvicorn:", e)

try:
    from fastapi import FastAPI
    print("✓ FastAPI imported successfully")
except ImportError as e:
    print("✗ Failed to import FastAPI:", e)

try:
    import main
    print("✓ main module imported successfully")
    print("✓ FastAPI app found:", hasattr(main, 'app'))
except ImportError as e:
    print("✗ Failed to import main:", e)
except Exception as e:
    print("✗ Error importing main:", e)

# Try to start a simple HTTP server
print("\nAttempting to start uvicorn...")
try:
    port = int(os.environ.get("PORT", 8000))
    print(f"Binding to 0.0.0.0:{port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
except Exception as e:
    print(f"✗ Failed to start server: {e}")
    import traceback
    traceback.print_exc()

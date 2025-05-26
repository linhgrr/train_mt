
import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Always bind to all interfaces for Render
    
    print(f"Starting server on {host}:{port}")
    print(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info"
    )

from fastapi import FastAPI
import uvicorn
import os

# Create a simple FastAPI app for testing
app = FastAPI(title="Test API")

@app.get("/")
async def root():
    """Simple test endpoint"""
    return {
        "message": "API is working!",
        "port": os.environ.get("PORT", "Not set")
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

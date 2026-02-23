"""
Frisbot FastAPI Backend with LLM Integration
Complete conversational AI companion with DeepSeek and SQLite.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from frisbot.api.chat import router as chat_router

# Create FastAPI app
app = FastAPI(
    title="Frisbot API",
    description="Conversational AI companion with Bayesian cognitive architecture",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",  # Vite default
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)


@app.get("/")
async def root():
    """Health check and status endpoint."""
    has_api_key = bool(os.getenv("DEEPSEEK_API_KEY"))
    
    return {
        "status": "online",
        "name": "Frisbot",
        "version": "0.1.0",
        "description": "Bayesian cognitive companion with DeepSeek LLM",
        "llm_configured": has_api_key,
        "database": "SQLite",
        "endpoints": {
            "chat": "/api/chat",
            "companions": "/api/companions",
            "health": "/"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "api": "online",
            "database": "connected",
            "llm": "configured" if os.getenv("DEEPSEEK_API_KEY") else "not configured"
        }
    }


# Run with: uvicorn main_llm:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n⚠️  Warning: DEEPSEEK_API_KEY not found in environment!")
        print("   Set it in .env file or export DEEPSEEK_API_KEY=your_key")
        print("   The API will run but LLM features will use fallbacks.\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
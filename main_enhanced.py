"""
Enhanced Frisbot with metacognitive capabilities.
Includes quality-based solvency and DeepSeek self-evaluation.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all routers
from frisbot.api.chat import router as chat_router
from frisbot.api.enhanced_chat import router as enhanced_router
from frisbot.api.metacognitive_chat import router as metacognitive_router

# Create FastAPI app
app = FastAPI(
    title="Frisbot Enhanced API",
    description="Conversational AI with Bayesian metacognitive architecture",
    version="0.2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:5173",
        "http://localhost:8080",
        "*"  # For testing - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(chat_router)  # Original at /api/chat
app.include_router(enhanced_router)  # Quality-aware at /api/v2/chat
app.include_router(metacognitive_router)  # Metacognitive at /api/metacognitive/chat


@app.get("/")
async def root():
    """Enhanced status endpoint."""
    has_api_key = bool(os.getenv("DEEPSEEK_API_KEY"))
    
    return {
        "status": "online",
        "name": "Frisbot Enhanced",
        "version": "0.2.0",
        "description": "Bayesian metacognitive companion with DeepSeek",
        "llm_configured": has_api_key,
        "database": "SQLite",
        "endpoints": {
            "original": {
                "chat": "/api/chat",
                "companions": "/api/companions"
            },
            "quality_enhanced": {
                "chat": "/api/v2/chat",
                "quality_metrics": "/api/v2/companions/{id}/quality"
            },
            "metacognitive": {
                "chat": "/api/metacognitive/chat",
                "metacognition": "/api/metacognitive/companions/{id}/metacognition",
                "introspection": "/api/metacognitive/companions/{id}/introspect",
                "belief_dynamics": "/api/metacognitive/companions/{id}/belief-dynamics"
            }
        },
        "features": {
            "basic": "Standard Bayesian belief updates",
            "enhanced": "Quality-based solvency adjustment",
            "metacognitive": "DeepSeek self-evaluation as Bayesian observations"
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
            "llm": "configured" if os.getenv("DEEPSEEK_API_KEY") else "not configured",
            "versions": {
                "basic": "active",
                "enhanced": "active",
                "metacognitive": "active"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n⚠️  Warning: DEEPSEEK_API_KEY not found!")
        print("   Set it in .env file or export DEEPSEEK_API_KEY=your_key")
        print("   The API will run but LLM features will use fallbacks.\n")
    else:
        print("\n✅ DeepSeek API key found")
        print("\n🚀 Starting Enhanced Frisbot with:")
        print("   - Basic chat at /api/chat")
        print("   - Quality-enhanced at /api/v2/chat")
        print("   - Metacognitive at /api/metacognitive/chat")
        print("\n📚 API docs at http://localhost:8001/docs\n")
    
    # Run on different port to not conflict with existing instance
    uvicorn.run(app, host="0.0.0.0", port=8001)
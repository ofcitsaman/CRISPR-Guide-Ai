"""
FastAPI application entrypoint for CRISPR Guide AI.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.endpoints import router as api_router
from backend.app.models.predictor import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to load trained models at startup."""
    print("Starting CRISPR Guide AI Backend Service...")
    try:
        predictor.load_artifacts()
        print("Models successfully loaded and ready for inference.")
    except Exception as e:
        print(f"Warning during model loading: {e}. Models will be loaded on demand if available.")
    yield
    print("Shutting down CRISPR Guide AI Backend Service.")


app = FastAPI(
    title="CRISPR Guide AI API",
    description="AI-Powered CRISPR-Cas9 Guide RNA Efficiency Prediction & Explainability Service",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For flexible development environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API router
app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {
        "service": "CRISPR Guide AI",
        "description": "Computational prediction of CRISPR-Cas9 single-guide RNA knockout efficiency",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

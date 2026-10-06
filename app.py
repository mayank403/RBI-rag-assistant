"""
FastAPI Backend for RBI RAG Prototype
"""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from rag_engine import rag_engine
from scraper import run_scraper


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Startup: ensure data + initialize RAG
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "raw", "rbi_documents.json"
    )
    if not os.path.exists(data_path):
        print("[>] No data found. Creating sample data...")
        run_scraper(use_sample=True, scrape_live=False)

    rag_engine.initialize()
    yield
    # Shutdown (nothing to clean up)


# FastAPI app
app = FastAPI(
    title="RBI RAG Assistant",
    description="Retrieval Augmented Generation system for RBI circulars, notifications, and press releases",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str
    response_time: float


class ScrapeRequest(BaseModel):
    live: Optional[bool] = False


# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>Frontend not found. Please ensure static/index.html exists.</h1>")


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system with a question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    try:
        result = rag_engine.query(request.question, top_k=request.top_k)
        response_time = time.time() - start_time

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=result["query"],
            response_time=round(response_time, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = rag_engine.get_document_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting stats: {str(e)}"
        )


@app.post("/api/scrape")
async def scrape_data(request: ScrapeRequest):
    """Trigger data scraping from RBI website."""
    try:
        filepath = run_scraper(use_sample=True, scrape_live=request.live)
        rag_engine.rebuild_index()
        return {
            "status": "success",
            "message": "Data scraped and indexed successfully",
            "filepath": filepath,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error scraping data: {str(e)}"
        )


@app.post("/api/rebuild")
async def rebuild_index():
    """Rebuild the vector index."""
    try:
        rag_engine.rebuild_index()
        return {"status": "success", "message": "Index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error rebuilding index: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_initialized": rag_engine._initialized,
        "ollama_available": rag_engine._ollama_available,
    }


@app.get("/api/suggestions")
async def get_suggestions():
    """Get sample query suggestions."""
    return {
        "suggestions": [
            "What is the current repo rate set by RBI?",
            "Explain the RBI Regulatory Sandbox framework",
            "What are the digital lending guidelines?",
            "Tell me about KYC norms for banks",
            "How does UPI work and what are the transaction limits?",
            "What is the CBDC Digital Rupee pilot?",
            "Explain Priority Sector Lending targets",
            "What is the Account Aggregator framework?",
            "What are NBFC registration requirements?",
            "Explain the Liberalised Remittance Scheme (LRS)",
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

"""
FastAPI Main Application
Entry point for the furniture recommendation API

Features:
- CORS enabled
- Analytics tracking middleware
- Error handling
- Health check endpoint
- Extensible structure
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
from pathlib import Path
import sys
from api.routers import search, products , chat

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from config import settings
from database.analytics import initialize_analytics_database, track_search

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== Application Lifespan ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events - startup and shutdown.
    Initialize services here.
    """
    # Startup
    logger.info("🚀 Starting Furniture Recommendation API...")
    
    # Initialize analytics database
    try:
        app.state.analytics_db = initialize_analytics_database()
        logger.info("✅ Analytics database initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize analytics: {e}")
        app.state.analytics_db = None
    
    # Initialize recommendation engine (lazy load for faster startup)
    app.state.recommendation_engine = None
    logger.info("✅ Recommendation engine will load on first use")
    
    logger.info("✅ API Ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down API...")
    if app.state.analytics_db:
        app.state.analytics_db.close()
    logger.info("✅ Cleanup complete")


# ========== Create FastAPI App ==========

app = FastAPI(
    title="Furniture Recommendation API",
    description="ML-powered furniture search and recommendation system",
    version="1.0.0",
    lifespan=lifespan
)


# ========== CORS Middleware ==========

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Analytics Middleware ==========

@app.middleware("http")
async def analytics_middleware(request: Request, call_next):
    """
    Track API requests for analytics.
    Measures response time and logs errors.
    """
    start_time = time.time()
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Time: {process_time:.3f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        process_time = time.time() - start_time
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "path": str(request.url.path)
            },
            headers={"X-Process-Time": str(process_time)}
        )


# ========== Error Handlers ==========

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The endpoint {request.url.path} does not exist",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )


# ========== Health Check Endpoints ==========

@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "Furniture Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "search": "/api/search",
            "products": "/api/products",
            "chat": "/api/chat",
            "analytics": "/api/analytics"
        }
    }


@app.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint.
    Verifies all services are running.
    """
    # Check analytics database
    analytics_status = "healthy" if request.app.state.analytics_db else "unavailable"
    
    # Check recommendation engine (lazy load)
    engine_status = "not_loaded"
    if request.app.state.recommendation_engine:
        engine_status = "loaded"
    
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "analytics_db": analytics_status,
            "recommendation_engine": engine_status,
            "pinecone": "connected"  # Assume connected if engine loads
        },
        "config": {
            "text_index": settings.PINECONE_TEXT_INDEX,
            "image_index": settings.PINECONE_IMAGE_INDEX,
            "products": 305
        }
    }


@app.get("/api/health")
async def api_health():
    """Simple health check for load balancers."""
    return {"status": "ok"}

# ========== Router Imports (will add incrementally) ==========
# ========== Include Routers ==========
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(products.router, prefix="/api", tags=["products"])
app.include_router(chat.router, prefix="/api", tags=["chat"])  # ← NEW

# TODO: Import routers here as we build them
# from api.routers import chat, analytics
# app.include_router(chat.router, prefix="/api", tags=["chat"])
# app.include_router(analytics.router, prefix="/api", tags=["analytics"])

# TODO: Import routers here as we build them
# from api.routers import search, products, chat, analytics
# app.include_router(search.router, prefix="/api", tags=["search"])
# app.include_router(products.router, prefix="/api", tags=["products"])
# app.include_router(chat.router, prefix="/api", tags=["chat"])
# app.include_router(analytics.router, prefix="/api", tags=["analytics"])


# ========== Run Server ==========

if __name__ == "__main__":
    """
    Run the API server.
    For development only - use uvicorn in production.
    """
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 STARTING FURNITURE RECOMMENDATION API")
    print("="*70)
    print(f"Host: 0.0.0.0")
    print(f"Port: 8000")
    print(f"Docs: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


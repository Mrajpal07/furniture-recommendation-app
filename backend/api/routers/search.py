"""
Search Router
Handles all search-related endpoints

Endpoints:
- POST /api/search/text - Text-based search
- POST /api/search/image - Image-based search  
- POST /api/search/hybrid - Hybrid search
- POST /api/search/generate-text-embedding - Generate embedding from text
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import List
import time
import logging
import json
from sentence_transformers import SentenceTransformer

from models.requests import (
    TextSearchRequest, 
    ImageSearchRequest, 
    HybridSearchRequest
)
from models.responses import SearchResponse, ProductResponse, ErrorResponse
from core.recommendation_engine import RecommendationEngine, SearchFilters, create_engine
from database.analytics import track_search

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global model cache (lazy load)
_text_model = None


# ========== Dependencies ==========

def get_recommendation_engine(request: Request) -> RecommendationEngine:
    """
    Get or create recommendation engine.
    Lazy loads on first use.
    """
    if request.app.state.recommendation_engine is None:
        logger.info("🔄 Loading recommendation engine...")
        request.app.state.recommendation_engine = create_engine()
        logger.info("✅ Recommendation engine loaded")
    
    return request.app.state.recommendation_engine


def get_text_model():
    """
    Get or load text embedding model.
    Lazy loads on first use.
    """
    global _text_model
    if _text_model is None:
        logger.info("🔄 Loading text embedding model...")
        _text_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Text model loaded")
    return _text_model


# ========== Helper Functions ==========

def create_search_filters(
    min_price: float = None,
    max_price: float = None,
    categories: List[str] = None,
    brands: List[str] = None
) -> SearchFilters:
    """Create SearchFilters object from request params."""
    return SearchFilters(
        min_price=min_price,
        max_price=max_price,
        categories=categories,
        brands=brands
    )


def track_search_event(
    request: Request,
    session_id: str,
    query_type: str,
    query_text: str = None,
    results_count: int = 0,
    filters: dict = None
):
    """Track search in analytics database."""
    try:
        if request.app.state.analytics_db:
            filters_json = json.dumps(filters) if filters else None
            track_search(
                request.app.state.analytics_db,
                session_id=session_id or "anonymous",
                query_text=query_text,
                query_type=query_type,
                filters_applied=filters_json,
                results_count=results_count
            )
    except Exception as e:
        logger.error(f"Failed to track search: {e}")


# ========== Endpoints ==========

@router.post(
    "/search/text",
    response_model=SearchResponse,
    summary="Text Search",
    description="Search products using text query with semantic understanding"
)
async def text_search(
    search_request: TextSearchRequest,
    request: Request,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Perform text-based semantic search.
    
    Uses Sentence Transformers to understand query meaning,
    not just keyword matching.
    """
    start_time = time.time()
    
    try:
        # Generate text embedding
        text_model = get_text_model()
        query_embedding = text_model.encode(search_request.query).tolist()
        
        # Create filters
        filters = create_search_filters(
            min_price=search_request.min_price,
            max_price=search_request.max_price,
            categories=search_request.categories,
            brands=search_request.brands
        )
        
        # Perform search
        results = engine.text_search(
            query_embedding=query_embedding,
            top_k=search_request.top_k,
            filters=filters
        )
        
        # Convert to response format
        product_responses = [
            ProductResponse(**result.to_dict())
            for result in results
        ]
        
        processing_time = time.time() - start_time
        
        # Track analytics
        filters_dict = {
            "min_price": search_request.min_price,
            "max_price": search_request.max_price,
            "categories": search_request.categories,
            "brands": search_request.brands
        }
        track_search_event(
            request=request,
            session_id=search_request.session_id,
            query_type="text",
            query_text=search_request.query,
            results_count=len(results),
            filters=filters_dict
        )
        
        return SearchResponse(
            success=True,
            results=product_responses,
            total_results=len(product_responses),
            query_info={
                "search_type": "text",
                "query": search_request.query,
                "filters_applied": filters_dict,
                "top_k": search_request.top_k
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SearchError",
                "message": f"Text search failed: {str(e)}"
            }
        )


@router.post(
    "/search/image",
    response_model=SearchResponse,
    summary="Image Search",
    description="Search products using image embedding (visual similarity)"
)
async def image_search(
    search_request: ImageSearchRequest,
    request: Request,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Perform image-based visual search.
    
    Requires pre-computed image embedding (512d CLIP vector).
    """
    start_time = time.time()
    
    try:
        # Validate embedding dimension
        if len(search_request.image_embedding) != 512:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ValidationError",
                    "message": f"Image embedding must be 512 dimensions, got {len(search_request.image_embedding)}"
                }
            )
        
        # Create filters
        filters = create_search_filters(
            min_price=search_request.min_price,
            max_price=search_request.max_price,
            categories=search_request.categories
        )
        
        # Perform search
        results = engine.image_search(
            query_embedding=search_request.image_embedding,
            top_k=search_request.top_k,
            filters=filters
        )
        
        # Convert to response format
        product_responses = [
            ProductResponse(**result.to_dict())
            for result in results
        ]
        
        processing_time = time.time() - start_time
        
        # Track analytics
        filters_dict = {
            "min_price": search_request.min_price,
            "max_price": search_request.max_price,
            "categories": search_request.categories
        }
        track_search_event(
            request=request,
            session_id=search_request.session_id,
            query_type="image",
            results_count=len(results),
            filters=filters_dict
        )
        
        return SearchResponse(
            success=True,
            results=product_responses,
            total_results=len(product_responses),
            query_info={
                "search_type": "image",
                "filters_applied": filters_dict,
                "top_k": search_request.top_k
            },
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SearchError",
                "message": f"Image search failed: {str(e)}"
            }
        )


@router.post(
    "/search/hybrid",
    response_model=SearchResponse,
    summary="Hybrid Search",
    description="Search using both text and image with custom weights"
)
async def hybrid_search(
    search_request: HybridSearchRequest,
    request: Request,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Perform hybrid search combining text and image.
    
    Allows custom weighting of text vs image similarity.
    """
    start_time = time.time()
    
    try:
        # Validate at least one input
        if not search_request.query and not search_request.image_embedding:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ValidationError",
                    "message": "Must provide either query text or image embedding"
                }
            )
        
        # Generate text embedding if query provided
        text_embedding = None
        if search_request.query:
            text_model = get_text_model()
            text_embedding = text_model.encode(search_request.query).tolist()
        
        # Validate image embedding if provided
        if search_request.image_embedding and len(search_request.image_embedding) != 512:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ValidationError",
                    "message": f"Image embedding must be 512 dimensions"
                }
            )
        
        # Create filters
        filters = create_search_filters(
            min_price=search_request.min_price,
            max_price=search_request.max_price,
            categories=search_request.categories
        )
        
        # Perform hybrid search
        results = engine.hybrid_search(
            text_embedding=text_embedding,
            image_embedding=search_request.image_embedding,
            text_weight=search_request.text_weight,
            image_weight=search_request.image_weight,
            top_k=search_request.top_k,
            filters=filters
        )
        
        # Convert to response format
        product_responses = [
            ProductResponse(**result.to_dict())
            for result in results
        ]
        
        processing_time = time.time() - start_time
        
        # Track analytics
        filters_dict = {
            "min_price": search_request.min_price,
            "max_price": search_request.max_price,
            "categories": search_request.categories
        }
        track_search_event(
            request=request,
            session_id=search_request.session_id,
            query_type="hybrid",
            query_text=search_request.query,
            results_count=len(results),
            filters=filters_dict
        )
        
        return SearchResponse(
            success=True,
            results=product_responses,
            total_results=len(product_responses),
            query_info={
                "search_type": "hybrid",
                "query": search_request.query,
                "text_weight": search_request.text_weight,
                "image_weight": search_request.image_weight,
                "filters_applied": filters_dict,
                "top_k": search_request.top_k
            },
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SearchError",
                "message": f"Hybrid search failed: {str(e)}"
            }
        )


@router.post(
    "/search/generate-embedding",
    summary="Generate Text Embedding",
    description="Generate embedding vector from text query (for frontend use)"
)
async def generate_text_embedding(query: str):
    """
    Generate text embedding from query.
    
    Useful for frontend to generate embeddings client-side.
    """
    try:
        text_model = get_text_model()
        embedding = text_model.encode(query).tolist()
        
        return {
            "success": True,
            "query": query,
            "embedding": embedding,
            "dimension": len(embedding)
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EmbeddingError",
                "message": f"Failed to generate embedding: {str(e)}"
            }
        )
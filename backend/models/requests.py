"""
Pydantic Models for API Requests and Responses

These define the structure of data flowing in/out of the API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ========== Enums ==========

class SearchType(str, Enum):
    """Search type enumeration."""
    TEXT = "text"
    IMAGE = "image"
    HYBRID = "hybrid"


class ViewSource(str, Enum):
    """Product view source enumeration."""
    SEARCH = "search"
    SIMILAR = "similar"
    RECOMMENDATION = "recommendation"
    CHAT = "chat"


# ========== Request Models ==========

class TextSearchRequest(BaseModel):
    """Request model for text search."""
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(10, description="Number of results", ge=1, le=50)
    min_price: Optional[float] = Field(None, description="Minimum price filter")
    max_price: Optional[float] = Field(None, description="Maximum price filter")
    categories: Optional[List[str]] = Field(None, description="Category filters")
    brands: Optional[List[str]] = Field(None, description="Brand filters")
    session_id: Optional[str] = Field(None, description="User session ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "modern blue sofa",
                "top_k": 10,
                "min_price": 100,
                "max_price": 500,
                "session_id": "user_123"
            }
        }


class ImageSearchRequest(BaseModel):
    """Request model for image search."""
    image_embedding: List[float] = Field(..., description="Image embedding vector (512d)")
    top_k: int = Field(10, description="Number of results", ge=1, le=50)
    min_price: Optional[float] = Field(None, description="Minimum price filter")
    max_price: Optional[float] = Field(None, description="Maximum price filter")
    categories: Optional[List[str]] = Field(None, description="Category filters")
    session_id: Optional[str] = Field(None, description="User session ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_embedding": [0.1, -0.2, 0.3],  # Truncated for example
                "top_k": 10,
                "session_id": "user_123"
            }
        }


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    query: Optional[str] = Field(None, description="Search query text")
    image_embedding: Optional[List[float]] = Field(None, description="Image embedding")
    text_weight: float = Field(0.5, description="Text similarity weight", ge=0, le=1)
    image_weight: float = Field(0.5, description="Image similarity weight", ge=0, le=1)
    top_k: int = Field(10, description="Number of results", ge=1, le=50)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    categories: Optional[List[str]] = None
    session_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "modern sofa",
                "text_weight": 0.6,
                "image_weight": 0.4,
                "top_k": 10
            }
        }


# ========== Response Models ==========

class ProductResponse(BaseModel):
    """Product data in search results."""
    product_id: str
    title: str
    brand: str
    price: float
    category: str
    description: str
    image_url: str
    score: float = Field(..., description="Similarity score (0-1)")
    match_type: str = Field(..., description="Search type used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "prod_123",
                "title": "Modern Blue Sofa",
                "brand": "IKEA",
                "price": 299.99,
                "category": "Living Room",
                "description": "Comfortable 3-seater sofa...",
                "image_url": "https://example.com/image.jpg",
                "score": 0.95,
                "match_type": "text"
            }
        }


class SearchResponse(BaseModel):
    """Response model for search endpoints."""
    success: bool = True
    results: List[ProductResponse]
    total_results: int
    query_info: Dict[str, Any]
    processing_time: float = Field(..., description="Time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "results": [],
                "total_results": 10,
                "query_info": {
                    "search_type": "text",
                    "query": "modern sofa",
                    "filters_applied": {}
                },
                "processing_time": 0.123
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "Invalid query parameters",
                "details": {}
            }
        }


# ========== Product Detail Models ==========

class ProductDetailResponse(BaseModel):
    """Detailed product information."""
    product_id: str
    title: str
    brand: str
    price: float
    price_bucket: str
    category: str
    description: str
    manufacturer: str
    material: str
    color: str
    country_of_origin: str
    dimensions: Dict[str, Optional[float]]
    images: List[str]
    image_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "prod_123",
                "title": "Modern Blue Sofa",
                "brand": "IKEA",
                "price": 299.99,
                "price_bucket": "Medium",
                "category": "Living Room",
                "description": "Comfortable 3-seater...",
                "manufacturer": "IKEA",
                "material": "Fabric",
                "color": "Blue",
                "country_of_origin": "Sweden",
                "dimensions": {
                    "length": 80.0,
                    "width": 35.0,
                    "height": 32.0,
                    "volume": 89600.0
                },
                "images": ["url1", "url2"],
                "image_count": 2
            }
        }


class SimilarProductsRequest(BaseModel):
    """Request for finding similar products."""
    product_id: str = Field(..., description="Reference product ID")
    search_type: SearchType = Field(SearchType.HYBRID, description="Search strategy")
    top_k: int = Field(10, description="Number of results", ge=1, le=50)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "prod_123",
                "search_type": "hybrid",
                "top_k": 10
            }
        }
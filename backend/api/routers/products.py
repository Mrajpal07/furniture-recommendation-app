"""
Products Router - FastAPI endpoints for furniture product operations
Handles product listing, filtering, details, and metadata retrieval
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional, Dict, Any
import pickle
import logging
from pathlib import Path as FilePath
from config import settings

# Import your Pydantic models (adjust import path as needed)
# from backend.models import Product, ProductDetail, ProductListResponse, CategoryResponse, BrandResponse

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(
    prefix="/products",
    tags=["products"],
    responses={404: {"description": "Not found"}},
)

# Global variable to cache loaded products
_products_cache: Optional[List[Dict[str, Any]]] = None


def load_products() -> List[Dict[str, Any]]:
    """
    Load products from pickle file with caching.
    
    Returns:
        List[Dict[str, Any]]: List of product dictionaries
        
    Raises:
        HTTPException: If file not found or loading fails
    """
    global _products_cache
    
    # Return cached data if available
    if _products_cache is not None:
        return _products_cache
    
    try:
        pickle_path = settings.DATA_FILE
        
        if not pickle_path.exists():
            logger.error(f"Products file not found at {pickle_path}")
            raise HTTPException(
                status_code=500,
                detail="Products data file not found. Please run data processing scripts."
            )
        
        with open(pickle_path, "rb") as f:
            products_df = pickle.load(f)
        
        # Convert DataFrame to list of dictionaries
        _products_cache = products_df.to_dict('records')
        logger.info(f"Loaded {len(_products_cache)} products from cache")
        
        return _products_cache
        
    except Exception as e:
        logger.error(f"Error loading products: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load products: {str(e)}"
        )


def filter_products(
    products: List[Dict[str, Any]],
    category: Optional[str] = None,
    brand: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    color: Optional[str] = None,
    material: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Apply filters to product list.
    
    Args:
        products: List of product dictionaries
        category: Filter by category (case-insensitive, partial match)
        brand: Filter by brand (case-insensitive, partial match)
        min_price: Minimum price filter
        max_price: Maximum price filter
        color: Filter by color (case-insensitive, partial match)
        material: Filter by material (case-insensitive, partial match)
        
    Returns:
        List[Dict[str, Any]]: Filtered products
    """
    filtered = products.copy()
    
    # Category filter
    if category:
        category_lower = category.lower()
        filtered = [
            p for p in filtered 
            if p.get('categories') and category_lower in str(p['categories']).lower()
        ]
    
    # Brand filter
    if brand:
        brand_lower = brand.lower()
        filtered = [
            p for p in filtered 
            if p.get('brand') and brand_lower in str(p['brand']).lower()
        ]
    
    # Price filters
    if min_price is not None:
        filtered = [
            p for p in filtered 
            if p.get('price') and extract_price(p['price']) >= min_price
        ]
    
    if max_price is not None:
        filtered = [
            p for p in filtered 
            if p.get('price') and extract_price(p['price']) <= max_price
        ]
    
    # Color filter
    if color:
        color_lower = color.lower()
        filtered = [
            p for p in filtered 
            if p.get('color') and color_lower in str(p['color']).lower()
        ]
    
    # Material filter
    if material:
        material_lower = material.lower()
        filtered = [
            p for p in filtered 
            if p.get('material') and material_lower in str(p['material']).lower()
        ]
    
    return filtered


def extract_price(price_str: str) -> float:
    """
    Extract numeric price from price string.
    
    Args:
        price_str: Price string (e.g., "$199.99", "₹1,499")
        
    Returns:
        float: Numeric price value
    """
    try:
        # Remove currency symbols, commas, and extract number
        clean_price = ''.join(c for c in str(price_str) if c.isdigit() or c == '.')
        return float(clean_price) if clean_price else 0.0
    except (ValueError, TypeError):
        return 0.0


def paginate_results(
    items: List[Any],
    skip: int,
    limit: int
) -> Dict[str, Any]:
    """
    Paginate results and return with metadata.
    
    Args:
        items: List of items to paginate
        skip: Number of items to skip
        limit: Maximum items per page
        
    Returns:
        Dict with paginated data and metadata
    """
    total = len(items)
    paginated_items = items[skip:skip + limit]
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": paginated_items,
        "has_more": skip + limit < total
    }


# ==================== ENDPOINTS ====================

@router.get("/", response_model=Dict[str, Any])
async def get_products(
    skip: int = Query(0, ge=0, description="Number of products to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum products to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    color: Optional[str] = Query(None, description="Filter by color"),
    material: Optional[str] = Query(None, description="Filter by material"),
):
    """
    Get paginated list of products with optional filters.
    
    **Example:**
    ```
    GET /products?limit=10&category=chair&min_price=100&max_price=500
    ```
    
    Returns:
        - total: Total number of matching products
        - skip: Current offset
        - limit: Items per page
        - items: List of products
        - has_more: Whether more results exist
    """
    try:
        # Load all products
        products = load_products()
        
        # Apply filters
        filtered_products = filter_products(
            products=products,
            category=category,
            brand=brand,
            min_price=min_price,
            max_price=max_price,
            color=color,
            material=material
        )
        
# Paginate results
        result = paginate_results(filtered_products, skip, limit)
        
        # Remove embedding fields before returning (they're numpy arrays and can't be serialized)
        clean_items = []
        for item in result['items']:
            clean_item = {k: v for k, v in item.items() 
                        if k not in ['text_embedding', 'image_embedding', 'combined_embedding']}
            clean_items.append(clean_item)
        
        result['items'] = clean_items
        
        logger.info(
            f"Retrieved {len(result['items'])} products "
            f"(total: {result['total']}, filters: "
            f"category={category}, brand={brand}, "
            f"price={min_price}-{max_price})"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_products: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve products: {str(e)}"
        )


@router.get("/categories", response_model=Dict[str, Any])
async def get_categories():
    """
    Get all unique product categories with counts.
    
    Returns:
        Dictionary with categories and their product counts
    """
    try:
        products = load_products()
        
        # Extract and count categories
        category_counts = {}
        for product in products:
            categories = product.get('categories', '')
            if categories:
                # Handle comma-separated categories
                cats = [c.strip() for c in str(categories).split(',')]
                for cat in cats:
                    if cat:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Sort by count (descending)
        sorted_categories = dict(
            sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "total": len(sorted_categories),
            "categories": sorted_categories
        }
        
    except Exception as e:
        logger.error(f"Error in get_categories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve categories: {str(e)}"
        )


@router.get("/brands", response_model=Dict[str, Any])
async def get_brands():
    """
    Get all unique brands with product counts.
    
    Returns:
        Dictionary with brands and their product counts
    """
    try:
        products = load_products()
        
        # Extract and count brands
        brand_counts = {}
        for product in products:
            brand = product.get('brand', '').strip()
            if brand:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        # Sort by count (descending)
        sorted_brands = dict(
            sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "total": len(sorted_brands),
            "brands": sorted_brands
        }
        
    except Exception as e:
        logger.error(f"Error in get_brands: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve brands: {str(e)}"
        )


@router.get("/colors", response_model=Dict[str, Any])
async def get_colors():
    """
    Get all unique colors with product counts.
    
    Returns:
        Dictionary with colors and their product counts
    """
    try:
        products = load_products()
        
        # Extract and count colors
        color_counts = {}
        for product in products:
            color = product.get('color', '').strip()
            if color and color.lower() != 'nan':
                color_counts[color] = color_counts.get(color, 0) + 1
        
        # Sort by count (descending)
        sorted_colors = dict(
            sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "total": len(sorted_colors),
            "colors": sorted_colors
        }
        
    except Exception as e:
        logger.error(f"Error in get_colors: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve colors: {str(e)}"
        )


@router.get("/materials", response_model=Dict[str, Any])
async def get_materials():
    """
    Get all unique materials with product counts.
    
    Returns:
        Dictionary with materials and their product counts
    """
    try:
        products = load_products()
        
        # Extract and count materials
        material_counts = {}
        for product in products:
            material = product.get('material', '').strip()
            if material and material.lower() != 'nan':
                material_counts[material] = material_counts.get(material, 0) + 1
        
        # Sort by count (descending)
        sorted_materials = dict(
            sorted(material_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "total": len(sorted_materials),
            "materials": sorted_materials
        }
        
    except Exception as e:
        logger.error(f"Error in get_materials: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve materials: {str(e)}"
        )


@router.get("/{product_id}", response_model=Dict[str, Any])
async def get_product_by_id(
    product_id: str = Path(..., description="Unique product identifier")
):
    """
    Get detailed information about a specific product.
    
    Args:
        product_id: The unique identifier of the product
        
    Returns:
        Complete product details
    """
    try:
        products = load_products()
        
        # Find product by uniq_id
        product = next(
            (p for p in products if str(p.get('uniq_id')) == product_id),
            None
        )
        
        if not product:
            raise HTTPException(
                status_code=404,
                detail=f"Product with ID '{product_id}' not found"
            )
        
        # Remove embedding fields from response (too large)
        response_product = {
            k: v for k, v in product.items() 
            if k not in ['text_embedding', 'image_embedding', 'combined_embedding']
        }
        
        logger.info(f"Retrieved product {product_id}")
        return response_product
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_product_by_id: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve product: {str(e)}"
        )


@router.get("/stats/overview", response_model=Dict[str, Any])
async def get_product_stats():
    """
    Get overview statistics about the product catalog.
    
    Returns:
        Statistics including counts, price ranges, etc.
    """
    try:
        products = load_products()
        
        # Extract prices
        prices = [extract_price(p.get('price', '0')) for p in products]
        valid_prices = [p for p in prices if p > 0]
        
        # Count categories
        unique_categories = set()
        for p in products:
            cats = str(p.get('categories', '')).split(',')
            unique_categories.update(c.strip() for c in cats if c.strip())
        
        # Count brands
        unique_brands = set(
            p.get('brand', '').strip() 
            for p in products 
            if p.get('brand', '').strip()
        )
        
        stats = {
            "total_products": len(products),
            "total_categories": len(unique_categories),
            "total_brands": len(unique_brands),
            "price_stats": {
                "min": min(valid_prices) if valid_prices else 0,
                "max": max(valid_prices) if valid_prices else 0,
                "avg": sum(valid_prices) / len(valid_prices) if valid_prices else 0,
            },
            "products_with_images": sum(
                1 for p in products if p.get('images')
            ),
            "products_with_description": sum(
                1 for p in products if p.get('description')
            ),
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_product_stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


# Health check endpoint
@router.get("/health/check")
async def health_check():
    """
    Check if products data is loaded and accessible.
    
    Returns:
        Health status of the products service
    """
    try:
        products = load_products()
        return {
            "status": "healthy",
            "products_loaded": len(products),
            "cache_active": _products_cache is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
"""
Core Recommendation Engine
Handles all search strategies: text, image, and hybrid

This is the brain of your recommendation system.
Connects to Pinecone and provides ranked product results.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from pinecone import Pinecone
import logging
from dataclasses import dataclass

# Import config
# Import config (fixed path)



# Add parent directory to Python path
import sys
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import settings  # Move this line here




# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== Data Models ==========

@dataclass
class SearchResult:
    """
    Represents a single search result.
    """
    product_id: str
    title: str
    brand: str
    price: float
    category: str
    description: str
    image_url: str
    score: float  # Similarity score
    match_type: str  # 'text', 'image', or 'hybrid'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            'product_id': self.product_id,
            'title': self.title,
            'brand': self.brand,
            'price': self.price,
            'category': self.category,
            'description': self.description[:200] + '...' if len(self.description) > 200 else self.description,
            'image_url': self.image_url,
            'score': round(self.score, 4),
            'match_type': self.match_type
        }


@dataclass
class SearchFilters:
    """
    Filters for search queries.
    """
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    categories: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    materials: Optional[List[str]] = None
    colors: Optional[List[str]] = None


# ========== Main Recommendation Engine ==========

class RecommendationEngine:
    """
    Core recommendation engine with multiple search strategies.
    
    Features:
    - Text-based semantic search
    - Image-based visual search
    - Hybrid search (text + image)
    - Advanced filtering
    - Smart ranking
    """
    
    def __init__(self):
        """Initialize the recommendation engine."""
        logger.info("Initializing Recommendation Engine...")
        
        # Load product data
        self.df = self._load_product_data()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.text_index = self.pc.Index(settings.PINECONE_TEXT_INDEX)
        self.image_index = self.pc.Index(settings.PINECONE_IMAGE_INDEX)
        
        logger.info(f"✅ Engine ready with {len(self.df)} products")
    
    
    def _load_product_data(self) -> pd.DataFrame:
        """
        Load product data with embeddings.
        
        Returns:
            DataFrame with all product data
        """
        try:
            df = pd.read_pickle(settings.DATA_FILE)
            logger.info(f"Loaded {len(df)} products from {settings.DATA_FILE}")
            return df
        except Exception as e:
            logger.error(f"Failed to load product data: {e}")
            raise
    
    
    def _apply_filters(self, 
                    results: List[Dict], 
                    filters: Optional[SearchFilters] = None) -> List[Dict]:
        """
        Apply filters to search results.
        
        Args:
            results: List of search results from Pinecone
            filters: SearchFilters object
            
        Returns:
            Filtered results
        """
        if not filters:
            return results
        
        filtered = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Price filter
            if filters.min_price and metadata.get('price', 0) < filters.min_price:
                continue
            if filters.max_price and metadata.get('price', float('inf')) > filters.max_price:
                continue
            
            # Category filter
            if filters.categories:
                product_category = metadata.get('category', '').lower()
                if not any(cat.lower() in product_category for cat in filters.categories):
                    continue
            
            # Brand filter
            if filters.brands:
                if metadata.get('brand', '').lower() not in [b.lower() for b in filters.brands]:
                    continue
            
            filtered.append(result)
        
        return filtered
    
    
    def _enrich_results(self, 
                    results: List[Dict],
                    match_type: str) -> List[SearchResult]:
        """
        Enrich Pinecone results with full product data.
        
        Args:
            results: Results from Pinecone
            match_type: Type of search ('text', 'image', 'hybrid')
            
        Returns:
            List of SearchResult objects
        """
        enriched = []
        
        for result in results:
            # Extract product ID (remove prefix like 'text_' or 'img_')
            full_id = result['id']
            product_id = full_id.replace('text_', '').replace('img_', '')
            
            # Get full product data
            product = self.df[self.df['uniq_id'] == product_id]
            
            if product.empty:
                logger.warning(f"Product {product_id} not found in dataframe")
                continue
            
            product = product.iloc[0]
            
            # Extract first image URL
            images = str(product.get('images', ''))
            image_url = images.split(',')[0].strip().strip("[]'\"") if images else ''
            
            # Create SearchResult
            search_result = SearchResult(
                product_id=product_id,
                title=product.get('title', 'Unknown'),
                brand=product.get('brand', 'Unknown'),
                price=float(product.get('price_numeric', 0)),
                category=product.get('categories', 'Unknown'),
                description=product.get('description', ''),
                image_url=image_url,
                score=result['score'],
                match_type=match_type
            )
            
            enriched.append(search_result)
        
        return enriched
    
    
    def text_search(self,
                query_embedding: List[float],
                top_k: int = 10,
                filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """
        Perform text-based semantic search.
        
        Args:
            query_embedding: Text embedding vector (384d)
            top_k: Number of results to return
            filters: Optional filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Text search: top_k={top_k}")
            
            # Query Pinecone text index
            results = self.text_index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more for filtering
                include_metadata=True
            )
            
            # Apply filters
            filtered = self._apply_filters(results['matches'], filters)
            
            # Limit to top_k after filtering
            filtered = filtered[:top_k]
            
            # Enrich with full product data
            enriched = self._enrich_results(filtered, match_type='text')
            
            logger.info(f"✅ Text search returned {len(enriched)} results")
            return enriched
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    
    def image_search(self,
                    query_embedding: List[float],
                    top_k: int = 10,
                    filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """
        Perform image-based visual search.
        
        Args:
            query_embedding: Image embedding vector (512d)
            top_k: Number of results to return
            filters: Optional filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"Image search: top_k={top_k}")
            
            # Query Pinecone image index
            results = self.image_index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more for filtering
                include_metadata=True
            )
            
            # Apply filters
            filtered = self._apply_filters(results['matches'], filters)
            
            # Limit to top_k after filtering
            filtered = filtered[:top_k]
            
            # Enrich with full product data
            enriched = self._enrich_results(filtered, match_type='image')
            
            logger.info(f"✅ Image search returned {len(enriched)} results")
            return enriched
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []
    
    
    def hybrid_search(self,
                    text_embedding: Optional[List[float]] = None,
                    image_embedding: Optional[List[float]] = None,
                    text_weight: float = 0.5,
                    image_weight: float = 0.5,
                    top_k: int = 10,
                    filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining text and image.
        
        Args:
            text_embedding: Text embedding vector (384d)
            image_embedding: Image embedding vector (512d)
            text_weight: Weight for text similarity (0-1)
            image_weight: Weight for image similarity (0-1)
            top_k: Number of results to return
            filters: Optional filters
            
        Returns:
            List of SearchResult objects ranked by combined score
        """
        try:
            logger.info(f"Hybrid search: text_weight={text_weight}, image_weight={image_weight}")
            
            results_dict = {}  # product_id -> combined_score
            
            # Text search
            if text_embedding:
                text_results = self.text_search(
                    query_embedding=text_embedding,
                    top_k=top_k * 2,
                    filters=filters
                )
                
                for result in text_results:
                    results_dict[result.product_id] = {
                        'text_score': result.score,
                        'image_score': 0.0,
                        'result': result
                    }
            
            # Image search
            if image_embedding:
                image_results = self.image_search(
                    query_embedding=image_embedding,
                    top_k=top_k * 2,
                    filters=filters
                )
                
                for result in image_results:
                    if result.product_id in results_dict:
                        results_dict[result.product_id]['image_score'] = result.score
                    else:
                        results_dict[result.product_id] = {
                            'text_score': 0.0,
                            'image_score': result.score,
                            'result': result
                        }
            
            # Calculate combined scores
            hybrid_results = []
            
            for product_id, data in results_dict.items():
                combined_score = (
                    data['text_score'] * text_weight +
                    data['image_score'] * image_weight
                )
                
                result = data['result']
                result.score = combined_score
                result.match_type = 'hybrid'
                
                hybrid_results.append(result)
            
            # Sort by combined score
            hybrid_results.sort(key=lambda x: x.score, reverse=True)
            
            # Return top_k
            final_results = hybrid_results[:top_k]
            
            logger.info(f"✅ Hybrid search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    
    def find_similar(self,
                    product_id: str,
                    search_type: str = 'text',
                    top_k: int = 10,
                    filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """
        Find similar products to a given product.
        
        Args:
            product_id: ID of the reference product
            search_type: 'text', 'image', or 'hybrid'
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            List of similar products
        """
        try:
            # Get product data
            product = self.df[self.df['uniq_id'] == product_id]
            
            if product.empty:
                logger.error(f"Product {product_id} not found")
                return []
            
            product = product.iloc[0]
            
            # Get embeddings
            text_emb = product.get('text_embedding')
            image_emb = product.get('image_embedding')
            
            # Convert to list if needed
            if hasattr(text_emb, 'tolist'):
                text_emb = text_emb.tolist()
            if hasattr(image_emb, 'tolist'):
                image_emb = image_emb.tolist()
            
            # Perform search based on type
            if search_type == 'text':
                results = self.text_search(text_emb, top_k=top_k+1, filters=filters)
            elif search_type == 'image':
                results = self.image_search(image_emb, top_k=top_k+1, filters=filters)
            elif search_type == 'hybrid':
                results = self.hybrid_search(
                    text_embedding=text_emb,
                    image_embedding=image_emb,
                    top_k=top_k+1,
                    filters=filters
                )
            else:
                logger.error(f"Invalid search_type: {search_type}")
                return []
            
            # Remove the reference product itself
            results = [r for r in results if r.product_id != product_id]
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Find similar failed: {e}")
            return []
    
    
    def get_product_details(self, product_id: str) -> Optional[Dict]:
        """
        Get detailed information about a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            Dictionary with product details
        """
        try:
            product = self.df[self.df['uniq_id'] == product_id]
            
            if product.empty:
                return None
            
            product = product.iloc[0]
            
            # Extract image URLs
            images = str(product.get('images', ''))
            image_urls = [
                url.strip().strip("[]'\"") 
                for url in images.split(',') 
                if url.strip()
            ]
            
            return {
                'product_id': product_id,
                'title': product.get('title', 'Unknown'),
                'brand': product.get('brand', 'Unknown'),
                'price': float(product.get('price_numeric', 0)),
                'price_bucket': product.get('price_bucket', 'Unknown'),
                'category': product.get('categories', 'Unknown'),
                'description': product.get('description', ''),
                'manufacturer': product.get('manufacturer', 'Unknown'),
                'material': product.get('material', 'Unknown'),
                'color': product.get('color', 'Unknown'),
                'country_of_origin': product.get('country_of_origin', 'Unknown'),
                'dimensions': {
                    'length': product.get('length'),
                    'width': product.get('width'),
                    'height': product.get('height'),
                    'volume': product.get('volume')
                },
                'images': image_urls,
                'image_count': len(image_urls)
            }
            
        except Exception as e:
            logger.error(f"Get product details failed: {e}")
            return None


# ========== Utility Functions ==========

def create_engine() -> RecommendationEngine:
    """
    Factory function to create recommendation engine instance.
    
    Returns:
        RecommendationEngine instance
    """
    return RecommendationEngine()


# ========== Testing ==========

if __name__ == "__main__":
    """
    Test the recommendation engine.
    """
    print("\n" + "="*70)
    print("🧪 TESTING RECOMMENDATION ENGINE")
    print("="*70)
    
    try:
        # Initialize engine
        engine = create_engine()
        
        # Test 1: Get a sample product
        print("\n📦 Test 1: Get Product Details")
        sample_id = engine.df['uniq_id'].iloc[0]
        details = engine.get_product_details(sample_id)
        print(f"Product: {details['title'][:50]}...")
        print(f"Price: ${details['price']:.2f}")
        print(f"Brand: {details['brand']}")
        
        # Test 2: Text search
        print("\n🔍 Test 2: Text Search (using sample product embedding)")
        sample_text_emb = engine.df['text_embedding'].iloc[0]
        if hasattr(sample_text_emb, 'tolist'):
            sample_text_emb = sample_text_emb.tolist()
        
        text_results = engine.text_search(sample_text_emb, top_k=5)
        print(f"Found {len(text_results)} results:")
        for i, result in enumerate(text_results, 1):
            print(f"  {i}. {result.title[:40]}... (${result.price:.2f}, score: {result.score:.4f})")
        
        # Test 3: Image search
        print("\n🖼️  Test 3: Image Search (using sample product embedding)")
        sample_image_emb = engine.df['image_embedding'].iloc[0]
        if hasattr(sample_image_emb, 'tolist'):
            sample_image_emb = sample_image_emb.tolist()
        
        image_results = engine.image_search(sample_image_emb, top_k=5)
        print(f"Found {len(image_results)} results:")
        for i, result in enumerate(image_results, 1):
            print(f"  {i}. {result.title[:40]}... (${result.price:.2f}, score: {result.score:.4f})")
        
        # Test 4: Hybrid search
        print("\n🎯 Test 4: Hybrid Search")
        hybrid_results = engine.hybrid_search(
            text_embedding=sample_text_emb,
            image_embedding=sample_image_emb,
            text_weight=0.6,
            image_weight=0.4,
            top_k=5
        )
        print(f"Found {len(hybrid_results)} results:")
        for i, result in enumerate(hybrid_results, 1):
            print(f"  {i}. {result.title[:40]}... (${result.price:.2f}, score: {result.score:.4f})")
        
        # Test 5: Find similar
        print("\n🔗 Test 5: Find Similar Products")
        similar_results = engine.find_similar(
            product_id=sample_id,
            search_type='hybrid',
            top_k=5
        )
        print(f"Found {len(similar_results)} similar products:")
        for i, result in enumerate(similar_results, 1):
            print(f"  {i}. {result.title[:40]}... (${result.price:.2f}, score: {result.score:.4f})")
        
        # Test 6: Search with filters
        print("\n🔍 Test 6: Search with Price Filter")
        filters = SearchFilters(min_price=20, max_price=100)
        filtered_results = engine.text_search(
            sample_text_emb,
            top_k=5,
            filters=filters
        )
        print(f"Found {len(filtered_results)} results in $20-$100 range:")
        for i, result in enumerate(filtered_results, 1):
            print(f"  {i}. {result.title[:40]}... (${result.price:.2f})")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nRecommendation Engine is ready for FastAPI integration! 🚀")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
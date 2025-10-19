"""
STEP 3: PINECONE SETUP AND EMBEDDING UPLOAD
Delete incorrect index, create proper indexes, upload all embeddings

This creates TWO indexes:
1. furniture-text (384 dims) - for text search
2. furniture-images (512 dims) - for visual search
"""

import pandas as pd
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path
from typing import List, Dict
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ========== Configuration ==========
DATA_FILE = Path('data/processed/data_with_all_embeddings.pkl')

# Pinecone settings
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OLD_INDEX_NAME = 'furniture-embeddings'  # Your current 768-dim index
TEXT_INDEX_NAME = 'furniture-text'       # New 384-dim index
IMAGE_INDEX_NAME = 'furniture-images'    # New 512-dim index

TEXT_DIM = 384
IMAGE_DIM = 512

REGION = 'us-east-1'  # Your region
METRIC = 'cosine'     # Similarity metric

BATCH_SIZE = 100      # Upload in batches


# ========== Pinecone Management ==========

def init_pinecone():
    """
    Initialize Pinecone client.
    
    Returns:
        Pinecone client instance
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment!")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✅ Pinecone client initialized")
    return pc


def list_indexes(pc: Pinecone):
    """
    List all existing indexes.
    """
    print("\n📋 Current Indexes:")
    indexes = pc.list_indexes()
    
    if not indexes:
        print("   No indexes found")
        return []
    
    for idx in indexes:
        print(f"   - {idx['name']} ({idx.get('dimension', 'N/A')} dims)")
    
    return [idx['name'] for idx in indexes]


def delete_index(pc: Pinecone, index_name: str):
    """
    Delete an index if it exists.
    
    Args:
        pc: Pinecone client
        index_name: Name of index to delete
    """
    existing = [idx['name'] for idx in pc.list_indexes()]
    
    if index_name in existing:
        print(f"\n🗑️  Deleting index: {index_name}")
        pc.delete_index(index_name)
        
        # Wait for deletion
        while index_name in [idx['name'] for idx in pc.list_indexes()]:
            print("   Waiting for deletion...", end='\r')
            time.sleep(1)
        
        print(f"✅ Deleted: {index_name}        ")
    else:
        print(f"ℹ️  Index '{index_name}' does not exist, skipping deletion")


def create_index(pc: Pinecone, index_name: str, dimension: int):
    """
    Create a new serverless index.
    
    Args:
        pc: Pinecone client
        index_name: Name for new index
        dimension: Embedding dimension
    """
    print(f"\n🔨 Creating index: {index_name} ({dimension} dims)")
    
    # Create serverless index
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud='aws',
            region=REGION
        )
    )
    
    # Wait for index to be ready
    print("   Waiting for index to be ready...", end='')
    while not pc.describe_index(index_name).status['ready']:
        print(".", end='', flush=True)
        time.sleep(2)
    
    print("\n✅ Index created and ready!")


def verify_index(pc: Pinecone, index_name: str):
    """
    Verify index configuration.
    
    Args:
        pc: Pinecone client
        index_name: Index name to verify
    """
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    print(f"\n📊 Index Stats: {index_name}")
    print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
    print(f"   Dimension: {stats.get('dimension', 'N/A')}")
    print(f"   Index fullness: {stats.get('index_fullness', 0)}")


# ========== Data Processing ==========

def load_product_data(filepath: Path) -> pd.DataFrame:
    """
    Load product data with embeddings.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        DataFrame with products and embeddings
    """
    print("\n📂 Loading product data...")
    df = pd.read_pickle(filepath)
    print(f"✅ Loaded {len(df)} products")
    
    # Verify embeddings
    text_dims = len(df['text_embedding'].iloc[0])
    image_dims = len(df['image_embedding'].iloc[0])
    
    print(f"   Text embeddings: {text_dims} dims")
    print(f"   Image embeddings: {image_dims} dims")
    
    if text_dims != TEXT_DIM:
        raise ValueError(f"Text embedding dimension mismatch: {text_dims} != {TEXT_DIM}")
    if image_dims != IMAGE_DIM:
        raise ValueError(f"Image embedding dimension mismatch: {image_dims} != {IMAGE_DIM}")
    
    return df


def prepare_vectors_for_upload(df: pd.DataFrame, 
                               embedding_column: str,
                               id_prefix: str = '') -> List[Dict]:
    """
    Prepare vectors in Pinecone format.
    
    Args:
        df: DataFrame with products
        embedding_column: Column name containing embeddings
        id_prefix: Prefix for vector IDs (e.g., 'text_' or 'img_')
        
    Returns:
        List of vector dictionaries
    """
    vectors = []
    
    for idx, row in df.iterrows():
        # Create metadata (Pinecone has size limits, keep essential only)
        metadata = {
            'uniq_id': row['uniq_id'],
            'title': row['title'][:200],  # Truncate to avoid size limits
            'brand': row['brand'],
            'price': float(row['price_numeric']),
            'category': row['categories'][:100] if pd.notna(row['categories']) else '',
        }
        
        # Create vector entry
        embedding = row[embedding_column]
        # Convert to list if numpy array, otherwise use as-is
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        vector = {
            'id': f"{id_prefix}{row['uniq_id']}",
            'values': embedding,
            'metadata': metadata
        }
        
        vectors.append(vector)
    
    return vectors


def upload_vectors(pc: Pinecone, 
                  index_name: str,
                  vectors: List[Dict],
                  batch_size: int = BATCH_SIZE):
    """
    Upload vectors to Pinecone index in batches.
    
    Args:
        pc: Pinecone client
        index_name: Target index name
        vectors: List of vector dictionaries
        batch_size: Batch size for uploads
    """
    print(f"\n📤 Uploading to {index_name}...")
    print(f"   Total vectors: {len(vectors)}")
    print(f"   Batch size: {batch_size}")
    
    index = pc.Index(index_name)
    
    # Upload in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        
        progress = min(i + batch_size, len(vectors))
        print(f"   Uploaded: {progress}/{len(vectors)} ({progress/len(vectors)*100:.1f}%)", end='\r')
    
    print(f"\n✅ Upload complete!")
    
    # Wait for index to be updated
    print("   Waiting for index to update...", end='')
    time.sleep(5)  # Give Pinecone time to process
    print(" Done!")


def test_search(pc: Pinecone, 
               index_name: str,
               query_vector: List[float],
               top_k: int = 3) -> List[Dict]:
    """
    Test similarity search on index.
    
    Args:
        pc: Pinecone client
        index_name: Index to search
        query_vector: Query embedding
        top_k: Number of results
        
    Returns:
        Search results
    """
    index = pc.Index(index_name)
    
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    return results['matches']


# ========== Main Workflow ==========

def main():
    """
    Complete Pinecone setup workflow:
    1. Delete old 768-dim index
    2. Create new 384-dim and 512-dim indexes
    3. Upload text and image embeddings
    4. Verify and test
    """
    print("\n" + "="*70)
    print("🚀 STEP 3: PINECONE SETUP & UPLOAD")
    print("="*70)
    
    try:
        # Initialize Pinecone
        pc = init_pinecone()
        
        # Show current indexes
        list_indexes(pc)
        
        # ===== STEP 1: DELETE OLD INDEX =====
        print("\n" + "="*70)
        print("STEP 3.1: CLEANUP")
        print("="*70)
        
        delete_index(pc, OLD_INDEX_NAME)
        
        # ===== STEP 2: CREATE NEW INDEXES =====
        print("\n" + "="*70)
        print("STEP 3.2: CREATE NEW INDEXES")
        print("="*70)
        
        create_index(pc, TEXT_INDEX_NAME, TEXT_DIM)
        create_index(pc, IMAGE_INDEX_NAME, IMAGE_DIM)
        
        # ===== STEP 3: LOAD DATA =====
        print("\n" + "="*70)
        print("STEP 3.3: LOAD PRODUCT DATA")
        print("="*70)
        
        df = load_product_data(DATA_FILE)
        
        # ===== STEP 4: UPLOAD TEXT EMBEDDINGS =====
        print("\n" + "="*70)
        print("STEP 3.4: UPLOAD TEXT EMBEDDINGS")
        print("="*70)
        
        text_vectors = prepare_vectors_for_upload(
            df, 
            embedding_column='text_embedding',
            id_prefix='text_'
        )
        
        upload_vectors(pc, TEXT_INDEX_NAME, text_vectors)
        verify_index(pc, TEXT_INDEX_NAME)
        
        # ===== STEP 5: UPLOAD IMAGE EMBEDDINGS =====
        print("\n" + "="*70)
        print("STEP 3.5: UPLOAD IMAGE EMBEDDINGS")
        print("="*70)
        
        image_vectors = prepare_vectors_for_upload(
            df,
            embedding_column='image_embedding',
            id_prefix='img_'
        )
        
        upload_vectors(pc, IMAGE_INDEX_NAME, image_vectors)
        verify_index(pc, IMAGE_INDEX_NAME)
        
        # ===== STEP 6: TEST SEARCHES =====
        print("\n" + "="*70)
        print("STEP 3.6: TESTING SEARCHES")
        print("="*70)
        
        # Test text search
        print("\n🔍 Testing TEXT search...")
        sample_text_vector = df['text_embedding'].iloc[0]
        if hasattr(sample_text_vector, 'tolist'):
            sample_text_vector = sample_text_vector.tolist()
        text_results = test_search(pc, TEXT_INDEX_NAME, sample_text_vector, top_k=3)
        
        print("   Top 3 results:")
        for i, result in enumerate(text_results, 1):
            print(f"   {i}. {result['metadata']['title'][:50]}...")
            print(f"      Score: {result['score']:.4f}")
        
        # Test image search
        print("\n🖼️  Testing IMAGE search...")
        sample_image_vector = df['image_embedding'].iloc[0]
        if hasattr(sample_image_vector, 'tolist'):
            sample_image_vector = sample_image_vector.tolist()
        image_results = test_search(pc, IMAGE_INDEX_NAME, sample_image_vector, top_k=3)
        
        print("   Top 3 results:")
        for i, result in enumerate(image_results, 1):
            print(f"   {i}. {result['metadata']['title'][:50]}...")
            print(f"      Score: {result['score']:.4f}")
        
        # ===== FINAL SUMMARY =====
        print("\n" + "="*70)
        print("✅ STEP 3 COMPLETE!")
        print("="*70)
        
        print("\n📊 Final Status:")
        print(f"   ✅ Text Index: {TEXT_INDEX_NAME}")
        print(f"      - Dimension: {TEXT_DIM}")
        print(f"      - Vectors: {len(text_vectors)}")
        
        print(f"   ✅ Image Index: {IMAGE_INDEX_NAME}")
        print(f"      - Dimension: {IMAGE_DIM}")
        print(f"      - Vectors: {len(image_vectors)}")
        
        print("\n🎯 Next Steps:")
        print("   1. Build recommendation engine (Step 4)")
        print("   2. Create FastAPI endpoints (Step 5)")
        print("   3. Integrate chatbot (Step 6)")
        
        print("\n💡 Test your indexes:")
        print(f"   Text search ready at: {TEXT_INDEX_NAME}")
        print(f"   Image search ready at: {IMAGE_INDEX_NAME}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
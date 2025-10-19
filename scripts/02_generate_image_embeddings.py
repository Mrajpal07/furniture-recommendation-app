"""
STEP 2: GENERATE IMAGE EMBEDDINGS USING CLIP
Visual search capability - find similar furniture by image

This uses CLIP (Contrastive Language-Image Pre-training):
- Joint text-image embeddings
- Zero-shot image understanding
- Perfect for visual similarity search
"""

import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Install required packages
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ImportError:
    print("📦 Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'transformers', 'torch', 'pillow', '-q'])
    from transformers import CLIPProcessor, CLIPModel
    import torch

# ========== Configuration ==========
DATA_DIR = Path('data/processed')
EMBEDDINGS_DIR = Path('data/embeddings')
INPUT_FILE = DATA_DIR / 'cleaned_data_with_embeddings.pkl'
OUTPUT_FILE = DATA_DIR / 'data_with_all_embeddings.pkl'
IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / 'image_embeddings.npy'

# CLIP Model
CLIP_MODEL = "openai/clip-vit-base-patch32"  # 512-dim embeddings, balanced speed/quality
# Alternative: "openai/clip-vit-large-patch14" (768-dim, better quality, slower)

# Processing settings
BATCH_SIZE = 8  # Process images in batches
MAX_RETRIES = 3  # Retry failed image downloads
TIMEOUT = 10     # URL request timeout (seconds)


# ========== Helper Functions ==========

def download_image(url: str, timeout: int = TIMEOUT) -> Optional[Image.Image]:
    """
    Download image from URL with error handling.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        PIL Image or None if failed
    """
    try:
        # Clean URL (remove spaces)
        url = url.strip()
        
        # Download image
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Load as PIL Image
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
        
    except Exception as e:
        return None


def extract_first_image_url(image_string: str) -> str:
    """
    Extract first image URL from the images column.
    Handles formats like: "['url1', 'url2']" or "url1, url2"
    
    Args:
        image_string: String containing image URLs
        
    Returns:
        First image URL
    """
    if not image_string or pd.isna(image_string):
        return ""
    
    # Convert to string
    image_string = str(image_string)
    
    # Remove brackets and quotes
    image_string = image_string.strip("[]'\"")
    
    # Split by comma and take first
    urls = [url.strip().strip("'\"") for url in image_string.split(',')]
    
    return urls[0] if urls else ""


def load_clip_model(model_name: str = CLIP_MODEL):
    """
    Load CLIP model and processor.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"🔄 Loading CLIP model: {model_name}")
    print("   (First time will download ~600MB, then cached locally)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    print(f"✅ Model loaded successfully")
    
    return model, processor, device


def generate_image_embedding(image: Image.Image, 
                            model: CLIPModel,
                            processor: CLIPProcessor,
                            device: str) -> np.ndarray:
    """
    Generate embedding for a single image using CLIP.
    
    Args:
        image: PIL Image
        model: CLIP model
        processor: CLIP processor
        device: Device to run on (cpu/cuda)
        
    Returns:
        Embedding vector (512-dim for base model)
    """
    with torch.no_grad():
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Generate embedding
        image_features = model.get_image_features(**inputs)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = image_features.cpu().numpy().squeeze()
    
    return embedding


def process_product_images(df: pd.DataFrame,
                        model: CLIPModel,
                        processor: CLIPProcessor,
                        device: str) -> Tuple[List[np.ndarray], List[bool]]:
    """
    Process all product images and generate embeddings.
    
    Args:
        df: DataFrame with product data
        model: CLIP model
        processor: CLIP processor
        device: Device to run on
        
    Returns:
        Tuple of (embeddings list, success flags)
    """
    print("\n" + "="*70)
    print("🖼️  PROCESSING PRODUCT IMAGES")
    print("="*70)
    
    embeddings = []
    success_flags = []
    
    total = len(df)
    successful = 0
    failed = 0
    
    for idx, row in df.iterrows():
        # Progress
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Processing: {idx + 1}/{total} ({(idx+1)/total*100:.1f}%)", end='\r')
        
        # Extract image URL
        image_url = extract_first_image_url(row['images'])
        
        if not image_url:
            # No image URL
            embeddings.append(np.zeros(512))  # Zero embedding as placeholder
            success_flags.append(False)
            failed += 1
            continue
        
        # Try to download and process image
        retry_count = 0
        image_processed = False
        
        while retry_count < MAX_RETRIES and not image_processed:
            try:
                # Download image
                image = download_image(image_url)
                
                if image is None:
                    raise Exception("Failed to download image")
                
                # Generate embedding
                embedding = generate_image_embedding(image, model, processor, device)
                embeddings.append(embedding)
                success_flags.append(True)
                successful += 1
                image_processed = True
                
            except Exception as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    # Failed after retries
                    embeddings.append(np.zeros(512))  # Zero embedding
                    success_flags.append(False)
                    failed += 1
    
    print(f"\nProcessing complete: {successful} successful, {failed} failed")
    
    return embeddings, success_flags


def verify_image_embeddings(embeddings: List[np.ndarray], 
                        success_flags: List[bool]) -> dict:
    """
    Verify quality of generated image embeddings.
    
    Args:
        embeddings: List of embedding vectors
        success_flags: Success flags for each embedding
        
    Returns:
        Dictionary with statistics
    """
    print("\n" + "="*70)
    print("✅ VERIFYING IMAGE EMBEDDINGS")
    print("="*70)
    
    embeddings_array = np.array(embeddings)
    
    # Basic stats
    total = len(embeddings)
    successful = sum(success_flags)
    failed = total - successful
    success_rate = (successful / total) * 100
    
    print(f"\n1. Generation Statistics:")
    print(f"   Total products: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    # Embedding quality
    if successful > 0:
        valid_embeddings = embeddings_array[success_flags]
        norms = np.linalg.norm(valid_embeddings, axis=1)
        
        print(f"\n2. Embedding Quality (successful only):")
        print(f"   Dimension: {valid_embeddings.shape[1]}")
        print(f"   Min norm: {norms.min():.4f}")
        print(f"   Max norm: {norms.max():.4f}")
        print(f"   Mean norm: {norms.mean():.4f}")
        print(f"   Std norm: {norms.std():.4f}")
        
        # Check for zero embeddings
        zero_count = np.sum(np.all(valid_embeddings == 0, axis=1))
        print(f"   Zero embeddings: {zero_count}")
    
    # Recommendations
    print(f"\n3. Quality Assessment:")
    if success_rate >= 95:
        print(f"   ✅ Excellent! {success_rate:.1f}% success rate")
    elif success_rate >= 85:
        print(f"   ✅ Good! {success_rate:.1f}% success rate")
    elif success_rate >= 70:
        print(f"   ⚠️  Acceptable: {success_rate:.1f}% success rate")
        print(f"       Consider checking failed URLs")
    else:
        print(f"   ❌ Low success rate: {success_rate:.1f}%")
        print(f"       Many images failed to process")
    
    return {
        'total': total,
        'successful': successful,
        'failed': failed,
        'success_rate': success_rate,
        'embedding_dim': embeddings_array.shape[1]
    }


def save_embeddings(df: pd.DataFrame, 
                embeddings: List[np.ndarray],
                success_flags: List[bool],
                output_pkl: Path,
                output_npy: Path):
    """
    Save image embeddings to files.
    
    Args:
        df: DataFrame with product data
        embeddings: List of image embeddings
        success_flags: Success flags
        output_pkl: Output pickle file path
        output_npy: Output numpy file path
    """
    print("\n" + "="*70)
    print("💾 SAVING EMBEDDINGS")
    print("="*70)
    
    # Create directories
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    
    # Add to DataFrame
    df['image_embedding'] = embeddings
    df['image_embedding_success'] = success_flags
    
    # Save complete DataFrame (text + image embeddings)
    df.to_pickle(output_pkl)
    print(f"✅ Saved complete data: {output_pkl}")
    print(f"   Size: {output_pkl.stat().st_size / (1024*1024):.2f} MB")
    
    # Save image embeddings separately (numpy array)
    embeddings_array = np.array(embeddings)
    np.save(output_npy, embeddings_array)
    print(f"✅ Saved image embeddings: {output_npy}")
    print(f"   Shape: {embeddings_array.shape}")
    print(f"   Size: {output_npy.stat().st_size / (1024*1024):.2f} MB")


def display_sample_results(df: pd.DataFrame, n: int = 3):
    """
    Display sample products with both text and image embeddings.
    
    Args:
        df: DataFrame with all embeddings
        n: Number of samples to display
    """
    print("\n" + "="*70)
    print("🔍 SAMPLE PRODUCTS WITH IMAGE EMBEDDINGS")
    print("="*70)
    
    for idx in range(min(n, len(df))):
        product = df.iloc[idx]
        
        print(f"\n{'─'*70}")
        print(f"Product {idx + 1}:")
        print(f"{'─'*70}")
        print(f"ID: {product['uniq_id']}")
        print(f"Title: {product['title'][:60]}...")
        print(f"Price: ${product['price_numeric']:.2f}")
        
        # Text embedding
        text_emb = product['text_embedding']
        print(f"\nText Embedding:")
        print(f"  Dimension: {len(text_emb)}")
        print(f"  Preview: {text_emb[:3]}...")
        
        # Image embedding
        img_emb = product['image_embedding']
        img_success = product['image_embedding_success']
        print(f"\nImage Embedding:")
        print(f"  Success: {'✅ Yes' if img_success else '❌ No'}")
        print(f"  Dimension: {len(img_emb)}")
        if img_success:
            print(f"  Preview: {img_emb[:3]}...")
            print(f"  Norm: {np.linalg.norm(img_emb):.4f}")
        else:
            print(f"  Status: Failed to process")


# ========== Main Execution ==========

def main():
    """
    Main workflow for generating image embeddings.
    """
    print("\n" + "="*70)
    print("🚀 STEP 2: GENERATE IMAGE EMBEDDINGS")
    print("="*70)
    print("\nThis will enable visual search (upload image → find similar products)")
    print(f"Using CLIP model: {CLIP_MODEL}")
    print(f"Expected time: ~5-10 minutes for 305 products\n")
    
    try:
        # Load data
        print("📂 Loading product data...")
        df = pd.read_pickle(INPUT_FILE)
        print(f"✅ Loaded {len(df)} products")
        
        # Load CLIP model
        model, processor, device = load_clip_model(CLIP_MODEL)
        
        # Process images
        embeddings, success_flags = process_product_images(
            df, model, processor, device
        )
        
        # Verify quality
        stats = verify_image_embeddings(embeddings, success_flags)
        
        # Save results
        save_embeddings(
            df, embeddings, success_flags,
            OUTPUT_FILE, IMAGE_EMBEDDINGS_FILE
        )
        
        # Display samples
        display_sample_results(df)
        
        # Final summary
        print("\n" + "="*70)
        print("✅ STEP 2 COMPLETE!")
        print("="*70)
        print(f"\nResults:")
        print(f"  ✅ Generated embeddings for {stats['successful']}/{stats['total']} products")
        print(f"  ✅ Success rate: {stats['success_rate']:.1f}%")
        print(f"  ✅ Image embedding dimension: {stats['embedding_dim']}")
        
        print(f"\nFiles created:")
        print(f"  1. {OUTPUT_FILE}")
        print(f"     (Complete data: text + image embeddings)")
        print(f"  2. {IMAGE_EMBEDDINGS_FILE}")
        print(f"     (Image embeddings only, numpy format)")
        
        print(f"\nNext Steps:")
        print(f"  1. Review success rate above")
        print(f"  2. If >90% success, proceed to Step 3 (Pinecone upload)")
        print(f"  3. If <90% success, check failed image URLs")
        
        return df, stats
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    df, stats = main()
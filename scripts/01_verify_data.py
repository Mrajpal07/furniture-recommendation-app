"""
STEP 1: DATA VERIFICATION & QUALITY CHECKS
Verify embeddings, check data quality, prepare for production

Run this first to ensure your data is clean and ready!
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
DATA_DIR = Path('data/processed')
EMBEDDINGS_FILE = DATA_DIR / 'cleaned_data_with_embeddings.pkl'
OUTPUT_REPORT = DATA_DIR / 'data_quality_report.json'

# Quality thresholds
MIN_EMBEDDING_NORM = 0.5  # Minimum L2 norm for valid embedding
MAX_EMBEDDING_NORM = 2.0  # Maximum L2 norm
MIN_TEXT_LENGTH = 10      # Minimum text length
MIN_PRODUCTS = 300        # Minimum number of products expected


# ========== Verification Functions ==========

def load_and_verify_file(filepath: Path) -> pd.DataFrame:
    """
    Load pickle file and perform basic checks.
    
    Returns:
        DataFrame with loaded data
    """
    print("="*70)
    print("📂 STEP 1.1: LOADING DATA")
    print("="*70)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"❌ File not found: {filepath}\n"
            f"   Current directory: {Path.cwd()}\n"
            f"   Expected location: {filepath.absolute()}"
        )
    
    # Check file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"✅ File found: {filepath.name}")
    print(f"   Size: {file_size_mb:.2f} MB")
    
    # Load data
    try:
        df = pd.read_pickle(filepath)
        print(f"✅ Data loaded successfully")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        return df
    except Exception as e:
        raise Exception(f"❌ Error loading pickle file: {str(e)}")


def verify_required_columns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Check if all required columns exist.
    
    Returns:
        Dictionary of column checks
    """
    print("\n" + "="*70)
    print("📋 STEP 1.2: VERIFYING COLUMNS")
    print("="*70)
    
    required_columns = {
        'uniq_id': 'Unique product identifier',
        'title': 'Product title',
        'description': 'Product description',
        'brand': 'Product brand',
        'price_numeric': 'Numeric price',
        'categories': 'Product categories',
        'images': 'Image URLs',
        'text_embedding': 'Text embeddings (384 dim)',
        'preprocessed_text': 'Cleaned text for search'
    }
    
    results = {}
    for col, description in required_columns.items():
        exists = col in df.columns
        status = "✅" if exists else "❌"
        print(f"{status} {col:20s} - {description}")
        results[col] = exists
    
    missing = [col for col, exists in results.items() if not exists]
    if missing:
        print(f"\n⚠️  WARNING: Missing columns: {', '.join(missing)}")
    else:
        print(f"\n✅ All required columns present!")
    
    return results


def verify_embeddings_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Verify embedding quality and consistency.
    
    Returns:
        Dictionary with embedding statistics
    """
    print("\n" + "="*70)
    print("🧮 STEP 1.3: VERIFYING EMBEDDINGS")
    print("="*70)
    
    if 'text_embedding' not in df.columns:
        print("❌ No text_embedding column found!")
        return {'status': 'missing'}
    
    embeddings = df['text_embedding'].tolist()
    
    # Check 1: All embeddings exist
    null_count = df['text_embedding'].isna().sum()
    print(f"1. Completeness Check:")
    print(f"   Total products: {len(df)}")
    print(f"   Valid embeddings: {len(df) - null_count}")
    print(f"   Missing embeddings: {null_count}")
    
    if null_count > 0:
        print(f"   ⚠️  WARNING: {null_count} products missing embeddings!")
    else:
        print(f"   ✅ All products have embeddings")
    
    # Check 2: Dimension consistency
    dimensions = [len(emb) for emb in embeddings if isinstance(emb, (list, np.ndarray))]
    unique_dims = set(dimensions)
    
    print(f"\n2. Dimension Check:")
    print(f"   Embedding dimensions: {unique_dims}")
    
    if len(unique_dims) == 1:
        dim = list(unique_dims)[0]
        print(f"   ✅ All embeddings have consistent dimension: {dim}")
    else:
        print(f"   ❌ CRITICAL: Inconsistent dimensions found!")
        return {'status': 'inconsistent_dimensions', 'dimensions': unique_dims}
    
    # Check 3: Value ranges (L2 norms)
    norms = [np.linalg.norm(emb) for emb in embeddings[:100]]  # Sample 100
    
    print(f"\n3. Value Range Check (L2 norms, sample of 100):")
    print(f"   Min norm: {min(norms):.4f}")
    print(f"   Max norm: {max(norms):.4f}")
    print(f"   Mean norm: {np.mean(norms):.4f}")
    print(f"   Std norm: {np.std(norms):.4f}")
    
    # Flag anomalies
    anomalies = sum(1 for n in norms if n < MIN_EMBEDDING_NORM or n > MAX_EMBEDDING_NORM)
    if anomalies > 0:
        print(f"   ⚠️  WARNING: {anomalies}/100 embeddings outside normal range")
    else:
        print(f"   ✅ All norms within acceptable range")
    
    # Check 4: Non-zero embeddings
    zero_embeddings = sum(1 for emb in embeddings if np.allclose(emb, 0))
    print(f"\n4. Zero Embedding Check:")
    print(f"   Zero embeddings: {zero_embeddings}")
    
    if zero_embeddings > 0:
        print(f"   ⚠️  WARNING: {zero_embeddings} zero embeddings found!")
    else:
        print(f"   ✅ No zero embeddings")
    
    return {
        'status': 'valid',
        'dimension': dim,
        'total_embeddings': len(embeddings),
        'missing': null_count,
        'zero_embeddings': zero_embeddings,
        'norm_stats': {
            'min': float(min(norms)),
            'max': float(max(norms)),
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms))
        }
    }


def verify_text_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Verify text data quality.
    
    Returns:
        Dictionary with text quality metrics
    """
    print("\n" + "="*70)
    print("📝 STEP 1.4: VERIFYING TEXT QUALITY")
    print("="*70)
    
    # Check preprocessed text
    if 'preprocessed_text' not in df.columns:
        print("⚠️  No preprocessed_text column - will need to generate")
        return {'status': 'missing'}
    
    # Text length distribution
    text_lengths = df['preprocessed_text'].str.len()
    
    print(f"1. Text Length Statistics:")
    print(f"   Min length: {text_lengths.min()}")
    print(f"   Max length: {text_lengths.max()}")
    print(f"   Mean length: {text_lengths.mean():.1f}")
    print(f"   Median length: {text_lengths.median():.1f}")
    
    # Check for too-short texts
    short_texts = (text_lengths < MIN_TEXT_LENGTH).sum()
    print(f"\n2. Quality Check:")
    print(f"   Texts < {MIN_TEXT_LENGTH} chars: {short_texts}")
    
    if short_texts > 0:
        print(f"   ⚠️  WARNING: {short_texts} products have very short text")
    else:
        print(f"   ✅ All texts have sufficient length")
    
    # Check for empty texts
    empty_texts = df['preprocessed_text'].str.strip().eq('').sum()
    print(f"   Empty texts: {empty_texts}")
    
    if empty_texts > 0:
        print(f"   ❌ CRITICAL: {empty_texts} products have empty text!")
    else:
        print(f"   ✅ No empty texts")
    
    return {
        'status': 'valid',
        'length_stats': {
            'min': int(text_lengths.min()),
            'max': int(text_lengths.max()),
            'mean': float(text_lengths.mean()),
            'median': float(text_lengths.median())
        },
        'short_texts': int(short_texts),
        'empty_texts': int(empty_texts)
    }


def verify_metadata_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Verify metadata completeness and quality.
    
    Returns:
        Dictionary with metadata quality metrics
    """
    print("\n" + "="*70)
    print("🏷️  STEP 1.5: VERIFYING METADATA")
    print("="*70)
    
    metadata_columns = ['uniq_id', 'title', 'brand', 'price_numeric', 
                    'categories', 'images']
    
    results = {}
    
    for col in metadata_columns:
        if col not in df.columns:
            print(f"❌ {col}: Column missing!")
            results[col] = {'status': 'missing'}
            continue
        
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100
        
        status = "✅" if missing == 0 else "⚠️ "
        print(f"{status} {col:20s}: {missing:3d} missing ({missing_pct:5.1f}%)")
        
        results[col] = {
            'missing': int(missing),
            'missing_pct': float(missing_pct)
        }
    
    # Check unique IDs
    if 'uniq_id' in df.columns:
        duplicates = df['uniq_id'].duplicated().sum()
        if duplicates > 0:
            print(f"\n❌ CRITICAL: {duplicates} duplicate IDs found!")
            results['uniq_id']['duplicates'] = int(duplicates)
        else:
            print(f"\n✅ All product IDs are unique")
    
    # Check price validity
    if 'price_numeric' in df.columns:
        invalid_prices = ((df['price_numeric'] <= 0) | 
                        (df['price_numeric'] > 10000)).sum()
        if invalid_prices > 0:
            print(f"⚠️  WARNING: {invalid_prices} products with suspicious prices")
            results['price_numeric']['invalid'] = int(invalid_prices)
    
    return results


def verify_images(df: pd.DataFrame) -> Dict[str, any]:
    """
    Verify image URLs and availability.
    
    Returns:
        Dictionary with image statistics
    """
    print("\n" + "="*70)
    print("🖼️  STEP 1.6: VERIFYING IMAGES")
    print("="*70)
    
    if 'images' not in df.columns:
        print("❌ No images column found!")
        return {'status': 'missing'}
    
    # Count products with images
    has_images = df['images'].notna() & (df['images'] != '')
    products_with_images = has_images.sum()
    
    print(f"Products with images: {products_with_images}/{len(df)}")
    
    # Sample image URLs
    sample_images = df[has_images]['images'].head(3)
    print(f"\nSample image URLs:")
    for idx, img in enumerate(sample_images, 1):
        img_str = str(img)[:80] + "..." if len(str(img)) > 80 else str(img)
        print(f"  {idx}. {img_str}")
    
    if products_with_images < len(df) * 0.9:  # Less than 90%
        print(f"\n⚠️  WARNING: Only {products_with_images}/{len(df)} products have images")
    else:
        print(f"\n✅ Good image coverage")
    
    return {
        'status': 'valid',
        'total_products': len(df),
        'with_images': int(products_with_images),
        'coverage_pct': float((products_with_images / len(df)) * 100)
    }


def generate_quality_report(df: pd.DataFrame, checks: Dict) -> Dict:
    """
    Generate comprehensive quality report.
    
    Returns:
        Complete quality report dictionary
    """
    print("\n" + "="*70)
    print("📊 STEP 1.7: GENERATING QUALITY REPORT")
    print("="*70)
    
    report = {
        'dataset_info': {
            'total_products': len(df),
            'total_columns': len(df.columns),
            'file_size_mb': 1.5,  # From your input
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'checks': checks,
        'recommendations': []
    }
    
    # Generate recommendations based on checks
    recommendations = []
    
    if checks.get('embeddings', {}).get('missing', 0) > 0:
        recommendations.append(
            "⚠️  Regenerate embeddings for products with missing values"
        )
    
    if checks.get('text', {}).get('empty_texts', 0) > 0:
        recommendations.append(
            "⚠️  Fill empty text fields or remove those products"
        )
    
    if checks.get('images', {}).get('coverage_pct', 100) < 90:
        recommendations.append(
            "⚠️  Consider adding more product images for better visual search"
        )
    
    if not recommendations:
        recommendations.append("✅ Data quality is excellent! Ready for production.")
    
    report['recommendations'] = recommendations
    
    # Print summary
    print("\n📋 QUALITY SUMMARY:")
    print(f"   Total Products: {len(df)}")
    print(f"   Embedding Dimension: {checks.get('embeddings', {}).get('dimension', 'N/A')}")
    print(f"   Products with Images: {checks.get('images', {}).get('with_images', 'N/A')}")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"   {rec}")
    
    return report


def save_quality_report(report: Dict, output_path: Path):
    """
    Save quality report to JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj
    
    report = convert_types(report)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Quality report saved to: {output_path}")


def display_sample_products(df: pd.DataFrame, n: int = 3):
    """
    Display sample products for manual verification.
    """
    print("\n" + "="*70)
    print("🔍 SAMPLE PRODUCTS FOR VERIFICATION")
    print("="*70)
    
    for idx in range(min(n, len(df))):
        product = df.iloc[idx]
        print(f"\n{'─'*70}")
        print(f"Product {idx + 1}:")
        print(f"{'─'*70}")
        print(f"ID: {product['uniq_id']}")
        print(f"Title: {product['title'][:60]}...")
        print(f"Brand: {product.get('brand', 'N/A')}")
        print(f"Price: ${product.get('price_numeric', 0):.2f}")
        print(f"Text Length: {len(str(product.get('preprocessed_text', '')))} chars")
        print(f"Embedding Dim: {len(product['text_embedding'])}")
        print(f"Embedding Preview: {product['text_embedding'][:5]}...")
        print(f"Has Images: {'Yes' if product.get('images') else 'No'}")


# ========== Main Execution ==========

def main():
    """
    Main verification workflow.
    """
    print("\n" + "="*70)
    print("🚀 DATA VERIFICATION & QUALITY CHECKS")
    print("="*70)
    print("\nThis script will verify your data is production-ready.")
    print("It checks: embeddings, text quality, metadata, and images.\n")
    
    try:
        # Step 1: Load data
        df = load_and_verify_file(EMBEDDINGS_FILE)
        
        # Step 2: Verify columns
        column_checks = verify_required_columns(df)
        
        # Step 3: Verify embeddings
        embedding_checks = verify_embeddings_quality(df)
        
        # Step 4: Verify text
        text_checks = verify_text_quality(df)
        
        # Step 5: Verify metadata
        metadata_checks = verify_metadata_quality(df)
        
        # Step 6: Verify images
        image_checks = verify_images(df)
        
        # Step 7: Generate report
        all_checks = {
            'columns': column_checks,
            'embeddings': embedding_checks,
            'text': text_checks,
            'metadata': metadata_checks,
            'images': image_checks
        }
        
        report = generate_quality_report(df, all_checks)
        
        # Step 8: Save report
        save_quality_report(report, OUTPUT_REPORT)
        
        # Step 9: Display samples
        display_sample_products(df)
        
        # Final summary
        print("\n" + "="*70)
        print("✅ VERIFICATION COMPLETE!")
        print("="*70)
        print(f"\nNext Steps:")
        print(f"1. Review the quality report: {OUTPUT_REPORT}")
        print(f"2. Check recommendations above")
        print(f"3. If all looks good, proceed to Step 2 (Image Embeddings)")
        print(f"4. If issues found, fix them first")
        
        return df, report
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED!")
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    df, report = main()
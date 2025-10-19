import requests
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_from_google_drive(file_id, destination):
    """Download file from Google Drive"""
    URL = "https://drive.google.com/file/d/1lDPG-vbXYGCR9cGDU8IbXSvFxCoDB5Lv/view?usp=drive_link"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Create directory if it doesn't exist
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    
    # Save file
    logger.info(f"Downloading data file to {destination}...")
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    
    logger.info(f"✅ Downloaded data file successfully!")

if __name__ == "__main__":
    file_id = os.getenv("DATA_FILE_ID")
    
    if not file_id:
        logger.error("❌ DATA_FILE_ID environment variable not set!")
        exit(1)
    
    # Destination path
    dest_path = "data/processed/data_with_all_embeddings.pkl"
    
    # Check if file already exists
    if Path(dest_path).exists():
        logger.info("✅ Data file already exists, skipping download")
    else:
        download_from_google_drive(file_id, dest_path)


# backend/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class Settings:
    # Pinecone settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_TEXT_INDEX: str = 'furniture-text'
    PINECONE_IMAGE_INDEX: str = 'furniture-images'
    
    # Groq settings (AI chatbot)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")  # ← MOVED INSIDE CLASS
    
    # Data paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_FILE = BASE_DIR / "data" / "processed" / "data_with_all_embeddings.pkl"
    
    # Server Settings
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', '8000'))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'

settings = Settings()
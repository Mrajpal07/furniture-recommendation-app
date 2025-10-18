from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    return index

if __name__ == "__main__":
    index = init_pinecone()
    print("Pinecone connected:", index.describe_index_stats())
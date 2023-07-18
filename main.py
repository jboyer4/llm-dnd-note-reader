import qdrant_client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
key = os.getenv("QDRANT_API_KEY")
# This is a cloud host, try local too
host = os.getenv("QDRANT_HOST")


client = qdrant_client.QdrantClient(url=host, api_key=key)

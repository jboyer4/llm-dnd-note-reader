import os
import qdrant_client
from qdrant_client import models
from langchain.vectorstores import qdrant
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
key = os.getenv("QDRANT_API_KEY")
# This is a cloud host, try local too
host = os.getenv("QDRANT_HOST")


client = qdrant_client.QdrantClient(url=host, api_key=key)
# Size is the dimension of the vector returned by the embedding model
# 768 for intructor-xl, 1536 for openai,
# 384 for sentence-transformers/all-MiniLM-L6-v2
# (hugging face: https://huggingface.co/sentence-transformers)
# See https://huggingface.co/blog/mteb for details
distanceType = models.Distance.COSINE  # default type is cosine
vectors_config = models.VectorParams(size=100, distance=distanceType)
client.recreate_collection(
    collection_name="hackathon_test", vectors_config=vectors_config
)

import os
import qdrant_client
from qdrant_client import models
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
key = os.getenv("QDRANT_API_KEY")
host = os.getenv("QDRANT_HOST")  # This is a cloud host, try local too

client = qdrant_client.QdrantClient(url=host, api_key=key)

# Size is the dimension of the vector returned by the embedding model
# 768 for intructor-xl, 1536 for openai,
# 384 for sentence-transformers/all-MiniLM-L6-v2
# (hugging face: https://huggingface.co/sentence-transformers)
# See https://huggingface.co/blog/mteb for details
distanceType = models.Distance.COSINE  # default type is cosine
vectors_config = models.VectorParams(size=768, distance=distanceType)
collection_name = "hackathon_test"
client.recreate_collection(
    collection_name=collection_name, vectors_config=vectors_config
)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def getChunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


with open("myData.txt", "r") as file:
    raw = file.read()
    chunks = getChunks(raw)

print(chunks)
print(len(chunks))
# https://qdrant.tech/documentation/integrations/langchain/?selector=aHRtbCA%2BIGJvZHkgPiBkaXY6bnRoLW9mLXR5cGUoMSkgPiBzZWN0aW9uID4gZGl2ID4gZGl2ID4gZGl2ID4gYXJ0aWNsZSA%2BIGgx
doc_store = Qdrant.from_texts(
    texts=chunks,
    embedding=embedding,
    url=host,
    api_key=key,
    collection_name=collection_name,
)

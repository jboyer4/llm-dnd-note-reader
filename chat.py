import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
import qdrant_client

# Load environment variables from .env file
load_dotenv()
key = os.getenv("QDRANT_API_KEY")
host = os.getenv("QDRANT_HOST")  # This is a cloud host, try local too
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
client = qdrant_client.QdrantClient(url=host, api_key=key)
doc_store = Qdrant(
    client=client,
    collection_name="hackathon_test",
    embeddings=embeddings,
)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=doc_store.as_retriever()
)

query = "Where did the reperations come from?"
response = qa.run(query)
print(response)

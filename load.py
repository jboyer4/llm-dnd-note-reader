import qdrant_client
from qdrant_client import models
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv


def load_text():
    load_dotenv()
    # key = os.getenv("QDRANT_API_KEY")
    # host = os.getenv("QDRANT_HOST")  # This is a cloud host, try local too
    # client = qdrant_client.QdrantClient(url=host, api_key=key)
    # Run "client" locally in memory so I don't have to keep the db up
    client = qdrant_client.QdrantClient(":memory:")
    embeddings_model = (
        "sentence-transformers/all-mpnet-base-v2"  # How do you choose a model?
    )
    distanceType = models.Distance.COSINE
    vectors_config = models.VectorParams(size=768, distance=distanceType)
    collection_name = "hackathon_test2"
    client.recreate_collection(
        collection_name=collection_name, vectors_config=vectors_config
    )

    def getChunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        return chunks

    with open("myData3.txt", "r") as file:
        raw = file.read()
        chunks = getChunks(raw)

    doc_store = Qdrant.from_texts(
        texts=chunks,
        embedding=HuggingFaceEmbeddings(model_name=embeddings_model),
        location=":memory:",
        collection_name=collection_name,
    )
    return doc_store

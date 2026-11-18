import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "medium_blog_1.txt"
INDEX_NAME = os.environ.get("INDEX_NAME")


def create_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if pc.has_index(INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists.")
        return

    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


if __name__ == "__main__":

    loader = UnstructuredLoader(
        file_path=FILE_PATH,
        chunking_strategy="basic",
        max_characters=10_000_000,
        encoding="UTF-8",
    )

    document = loader.load()

    print("Chunking...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"Created {len(texts)} chunks...")

    print("Embedding...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Creating Pinecone index...")
    create_index()

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        texts, embedding=embeddings, index_name=INDEX_NAME
    )
    print("Pinecone database populated.")

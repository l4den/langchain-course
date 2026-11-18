import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from langchain_course.logger import setup_logger

load_dotenv()

INDEX_NAME = os.environ.get("INDEX_NAME")
CRAWLING_URL = "https://docs.langchain.com/oss/python/langchain/overview"

logger = setup_logger()

# credentials needed for web crawling using Tavily
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embeddings-3-small",
    show_progress_bar=False,
    chunk_size=50,  # How many text objects to send per request. too high = rate limited, too low = very slow embedding
    retry_min_seconds=10,  # If embedding fails how long do we wait before retrying.
)

# chroma = Chomra(presist_directory='chroma_db', embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(
    max_depth=5, max_breadth=20, max_pages=1000
)  # Returns all available URLs from the page (sitemap)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    logger.info(f"Preparing to index {len(documents)} documents.")

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    logger.info(
        f"Created {len(batches)} batches with {batch_size} documents per batch from {len(documents)} documents."
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_document(batch)
            logger.info(f"Batch {batch_num} added to vector store")
        except Exception as e:
            logger.error(f"Failed to add batch {batch_num} to vector store.")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successfull = [1 for result in results if result is True]

    if successfull == len(batches):
        logger.info(f"All {len(batches)} batches added to vectorstore successfully.")
    else:
        logger.warning(
            f"{successfull}/{len(batches)} batches added to vectorstore successfully."
        )


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


async def main():

    # 1. Get Data
    logger.info(f"Tavily Crawl started on: {CRAWLING_URL}")

    res = tavily_crawl.invoke(
        {
            "url": CRAWLING_URL,
            "max_depth": 1,  # How far from the base URL can the crawler explore?
            "extract_depth": "advanced",  # Gets more data (tables, embeded docs, etc).
            "instructions": "content on ai agents",  # Filtering mechanisam which uses NLP to decide which page to scrape.
        }
    )

    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]

    logger.info(
        f"[green]Tavily Crawl succesfully crawled {len(all_docs)} documents.[/green]"
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

    logger.info(
        f"Created {len(split_docs)} chunks from {len(all_docs)} documents.[/green]"
    )

    create_index()

    await index_documents_async(split_docs, batch_size=500)

    logger.info("PIPLELINE COMPLETED\n")
    logger.info("SUMMARY:")
    # logger.info(f'URLs mapped: {len(site_map["results"])}')
    logger.info(f"Documents extracted: {len(all_docs)}")
    logger.info(f"Chunks created: {len(split_docs)}")


if __name__ == "__main__":
    asyncio.run(main())

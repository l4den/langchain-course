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
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,  # How many text objects to send per request. too high = rate limited, too low = very slow embedding
    max_retries=6,
    retry_min_seconds=2, # If embedding fails how long do we wait before retrying.
    retry_max_seconds=30,
)



tavily_extract = TavilyExtract()
tavily_map = TavilyMap(
    max_depth=5, max_breadth=20, max_pages=1000
)  # Returns all available URLs from the page (sitemap)
tavily_crawl = TavilyCrawl()


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


#create_index()
vectorstore = Chroma(persist_directory='chroma_db', embedding_function=embeddings)
#vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    logger.info(f"Preparing to index {len(documents)} documents.")

    batches = [
        documents[i : i + batch_size] 
        for i in range(0, len(documents), batch_size)
    ]

    logger.info(
        f"Created {len(batches)} batches with {batch_size} documents per batch from {len(documents)} documents."
    )

    semaphore = asyncio.Semaphore(2)

    async def add_batch(batch: List[Document], batch_num: int):
        async with semaphore:
            try:
                await vectorstore.aadd_documents(batch)
                logger.info(f"Batch {batch_num} added to vector store")
            except Exception as e:
                logger.exception(f"Failed to add batch {batch_num} to vector store.")
                return False
            return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(result is True for result in results)

    if successful == len(batches):
        logger.info(f"All {len(batches)} batches added to vectorstore successfully.")
    else:
        logger.warning(
            f"{successful}/{len(batches)} batches added to vectorstore successfully."
        )


def chunk_urls(urls: List[str], chunk_size: int) -> List[List[str]]:
    chunks = []

    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i : i + chunk_size])

    return chunks


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    try:
        docs = await tavily_extract.ainvoke(input={"urls": urls})

        logger.info(
            f"Tavily Extract extracted {len(docs.get('results', []))} documents from batch {batch_num}."
        )

        return docs

    except Exception as e:
        logger.error(f"Tavily Extract failed to extract batch {batch_num}.\n{e}")
        return []


async def async_exract(url_batches: List[List[str]]):

    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_pages = []

    failed_batches = 0

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Batch failed with exception\n{result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    logger.info(f"Tavily Extract complete. {len(all_pages)} pages extracted.")

    if failed_batches > 0:
        logger.info(
            f"[yellow]Tavily Extract failed to extract {len(all_pages)} batches.[/yellow]"
        )

    return all_pages


async def main():

    # 1. Getting all crawlable URls.
    logger.info(f"Tavily Crawl started on: {CRAWLING_URL}")

    site_map = tavily_map.invoke(CRAWLING_URL)

    logger.info(f"Tavily Map found {len(site_map['results'])} URLs.")

    # 2. Split URLs into batches
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=10)

    logger.info(
        f"Chunked {len(site_map['results'])} URLs into {len(url_batches)} batches."
    )

    all_docs = await async_exract(url_batches)

    logger.info(
        f"[green]Tavily Crawl succesfully crawled {len(all_docs)} documents.[/green]"
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

    logger.info(
        f"Created {len(split_docs)} chunks from {len(all_docs)} documents.[/green]"
    )

    await index_documents_async(split_docs, batch_size=500)

    logger.info("PIPLELINE COMPLETED\n")
    logger.info("SUMMARY:")
    logger.info(f'URLs mapped: {len(site_map["results"])}')
    logger.info(f"Documents extracted: {len(all_docs)}")
    logger.info(f"Chunks created: {len(split_docs)}")


if __name__ == "__main__":
    asyncio.run(main())

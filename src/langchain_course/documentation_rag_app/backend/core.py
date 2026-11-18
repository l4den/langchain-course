import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

INDEX_NAME = os.environ.get("INDEX_NAME")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,  # How many text objects to send per request. too high = rate limited, too low = very slow embedding
    max_retries=6,
    retry_min_seconds=2, # If embedding fails how long do we wait before retrying.
    retry_max_seconds=30,
)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

model = init_chat_model('gpt-4o-mini', model_provider='openai')


@tool(response_format='content_and_artifact')
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user querries about LangChain."""

    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4) # get the 4 most relevant documents

    serialized = "\n\n".join(
        (f'Source {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}')
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query:str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a question using retrieved documents.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    system_prompt = (
        "You are a helpful AI assistant that asnwers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find the relevant information before answering the question. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documents, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [{'role': 'user', 'content': query}]

    response = agent.invoke({'messages': messages})

    answer = response['messages'][-1].content

    context_docs = []

    for message in response['messages']:
        if isinstance(message, ToolMessage) and hasattr(message, 'artifact'):

            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        'answer': answer,
        'context': context_docs
    }


if __name__ == "__main__":
    result = run_llm(query="What are deep agents?")
    print(result)
import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

MODEL_NAME = "gpt-4o-mini"


embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model=MODEL_NAME)
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template("""
    Answer this question only based on the following context:
    
    {context}

    Question:
    {question}
    
    Provide a detailed answer:
    """)


def format_docs(docs):
    """Format recieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def answer_without_rag(query: str):
    """
    LLM call
    """
    print("=" * 60)
    print("Implementation 1: LLM with no RAG")
    print("=" * 60)
    print("\n")

    response = llm.invoke(query)

    return response.content


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without using LCEL.
    """
    print("\n\n")
    print("=" * 60)
    print("Implementation 2: RAG without LCEL")
    print("=" * 60)
    print("\n")

    docs = retriever.invoke(query)

    context = format_docs(docs)

    message = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(message)

    return response.content


def retrieval_chain_with_lcel():
    """
    Creates a retrieval chain usingLCEL

    Send the prompt template to the LLM and parse the output with StrOutputParser()

    """

    print("\n\n")
    print("=" * 60)
    print("Implementation 3: RAG with LCEL")
    print("=" * 60)
    print("\n")

    # RunnablePassthrough takes in a dictionary and returns the original dictionay filled in with new keys and values which are
    # a result of the methods that ran inside.
    # In this implementation we take the value of the key "question" and we use it in the retriever,
    # the documents are formated by format_docs. The formated docs are outputed as the value of the key "context".
    # The outputed dictionary looks like this {'question': question, 'context': formated_documents_string}

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


def main():

    question = "What is Pinecone in machine learning?"

    print(answer_without_rag(question))
    print(retrieval_chain_without_lcel(question))
    chain_with_lcel = retrieval_chain_with_lcel()
    print(chain_with_lcel.invoke({"question": question}))


if __name__ == "__main__":
    main()

from dotenv import load_dotenv

load_dotenv()

from typing import List

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from tavily import TavilyClient


class Source(BaseModel):
    """Shema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent resonse"""

    answer: str = Field(description="The agent's response to the query")
    srouces: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that seaches the internet
    Args:
        query: The query to seach for
    Return:
        The search results
    """
    print(f"Agent searched for:\n{query}\n")
    return tavily.search(query=query)


def search_jobs():

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    tools = [TavilySearch()]

    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in skopje on linkedin and list their details"
            )
        }
    )
    print(result)

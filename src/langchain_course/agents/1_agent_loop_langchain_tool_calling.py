from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"


@tool
def get_product_price(product: str) -> float:
    """Lookup the price of a product from the product catalog."""
    print(f"Agent ran get_product_price({product})")
    prices = {"laptop": 63000, "headphones": 3350, "keyboard": 5500}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available discount tiers are: bronze, sliver and gold.
    """
    print(f"Agent ran apply_discount({price}, {discount_tier})")
    discounts = {"bronze": 5, "silver": 12, "gold": 23}
    return round(price * (1 - discounts.get(discount_tier, 0) / 100), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"openai:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"User asks: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. Never guess or assume any product price. You must call get_product_price to get the product's price. \n"
                "2. Only call apply_discount after you receive the price from get_product_price. \n"
                "Pass the exact price returned by get_product_price - do not pass a made-up number. \n"
                "3. Never calculate discounts yourself using math. Always use the apply_discount tool. \n"
                "4. If the user does not specify a discount tier ask them which tier to use - do not assume one. "
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}/{MAX_ITERATIONS}")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        tool_to_use = tools_dict.get(tool_name)

        print(f"Tool Selected = {tool_name}\nTool Args = {tool_args}")

        if tool_to_use is None:
            raise ValueError(f"Invalid tool selected. Tool={tool_name}")

        observation = tool_to_use.invoke(tool_args)

        print(f"Tool Result = {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached with no answer")
    return None


if __name__ == "__main__":
    print("Running LangChain Agent")
    result = run_agent(
        "What is the price of the product named laptop after applying the gold discount?"
    )

import json

from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Lookup the price of a product from the product catalog."""
    print(f"Agent ran get_product_price({product})")
    prices = {"laptop": 63000, "headphones": 3350, "keyboard": 5500}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available discount tiers are: bronze, sliver and gold.
    """
    print(f"Agent ran apply_discount({price}, {discount_tier})")
    discounts = {"bronze": 5, "silver": 12, "gold": 23}
    return round(price * (1 - discounts.get(discount_tier, 0) / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Retrieve the price of a product by its name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product.",
                    }
                },
                "required": ["product"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": (
                "Apply a discount tier to a price and return the final price. "
                "Available discount tiers are bronze, silver, and gold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price of the product.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "enum": ["bronze", "silver", "gold"],
                        "description": "The discount tier to apply.",
                    },
                },
                "required": ["price", "discount_tier"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

openai = OpenAI()


@traceable(name="LangChain OpenAI Agent Loop", run_type="llm")
def openai_chat_traceable(messages):
    return openai.chat.completions.create(
        model=MODEL, tools=tools_for_llm, messages=messages
    )


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"User asks: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. Never guess or assume any product price. You must call get_product_price to get the product's price.\n"
                "2. Only call apply_discount after you receive the price from get_product_price. "
                "Pass the exact price returned by get_product_price - do not pass a made-up number.\n"
                "3. Never calculate discounts yourself using math. Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, ask them which tier to use - do not assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}/{MAX_ITERATIONS}")

        response = openai_chat_traceable(messages)
        ai_message = response.choices[0].message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_call_id = tool_call.id

        tool_to_use = tools_dict.get(tool_name)

        print(f"Tool Selected = {tool_name}\nTool Args = {tool_args}")

        if tool_to_use is None:
            raise ValueError(f"Invalid tool selected. Tool={tool_name}")

        observation = tool_to_use(**tool_args)

        print(f"Tool Result = {observation}")

        messages.append(ai_message)
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": str(observation)}
        )

    print("ERROR: Max iterations reached with no answer")
    return None


if __name__ == "__main__":
    print("Running LangChain Agent")
    result = run_agent(
        "What is the price of the product named laptop with the gold tier discount?"
    )

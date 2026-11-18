import inspect
import json
import re

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

    price = float(price.strip("(").strip(")"))

    discounts = {"bronze": 5, "silver": 12, "gold": 23}
    return round(price * (1 - discounts.get(discount_tier, 0) / 100), 2)


def get_tool_descriptions(tool_dict):
    descriptions = []

    for tool_name, tool_function in tool_dict.items():

        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        doc_string = inspect.getdoc(tool_function) or ""

        descriptions.append(f"{tool_name}{signature} - {doc_string}")

    return "\n".join(descriptions)


tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. Never guess or assume any product price. You must call get_product_price to get the product's price
2. Only call apply_discount after you receive the price from get_product_price.Pass the exact price returned by get_product_price - do not pass a made-up number.
3. Never calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use - do not assume one.

Answer the user's question using the available tools.

Tools:
{tool_descriptions}

Follow this format:
Question: the user's question
Thought: decide what to do next
Action: select a tool from [{tool_names}]
Action Input: the arguments for the tool
Observation: the tool's result
Repeat the Thought/Action/Action Input/Observation
cycle as necessary.
Thought: I have enough information to answer.
Final Answer: the answer to the user.

Begin!

Question: {{question}}
Thought:"""

openai = OpenAI()


@traceable(name="LangChain OpenAI Agent Loop", run_type="llm")
def openai_chat_traceable(messages, options=None):
    return openai.chat.completions.create(
        model=MODEL, messages=messages, **(options or {})
    )


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    print(f"User asks: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}/{MAX_ITERATIONS}")

        full_prompt = prompt + scratchpad

        response = openai_chat_traceable(
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.choices[0].message.content

        print(f"Raw LLM output:\n{output}")

        final_answer_match = re.search(r"\nFinal Answer:\s*(.+)", output)

        if final_answer_match:
            return final_answer_match.group(1).strip()

        action_match = re.search(r"\nAction:\s*(.+)", output)
        action_input_match = re.search(r"\nAction Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("[ERROR] Invalid tool format")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        # tool_call_id = tool_call.id

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("\\").strip('"') for x in raw_args]

        # tool_to_use = tools_dict.get(tool_name)

        print(f"Tool Selected = {tool_name}\nTool Args = {args}")

        if tool_name not in tools:
            observation = (
                f"Error. Tool was not found. Available tools {list[str](tools.keys())}"
            )
        else:
            observation = str(tools[tool_name](*args))

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached with no answer")
    return None


if __name__ == "__main__":
    print("Running LangChain Agent")
    result = run_agent(
        "What is the price of the product named laptop with the gold tier discount?"
    )
    print(f"Agent answer: {result}")

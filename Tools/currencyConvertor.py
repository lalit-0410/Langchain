from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
from typing import Annotated
import requests

load_dotenv()

model = ChatOpenAI(model='gpt-4')


@tool
def get_currency_factor(base_currency: str, target_currency: str) -> float:
    """Give current conversion rate between two currencies."""

    base_currency = base_currency.lower()
    target_currency = target_currency.lower()

    url = (
        f"https://cdn.jsdelivr.net/npm/"
        f"@fawazahmed0/currency-api@latest/"
        f"v1/currencies/{base_currency}.json"
    )

    response = requests.get(url)
    data = response.json()

    return data[base_currency][target_currency]


result = get_currency_factor.invoke({
    "base_currency": "USD",
    "target_currency": "INR"
})

print(result)


@tool
def convertCurrency(
    base_currency_value: int,
    conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """This function gives the total amount."""

    return base_currency_value * conversion_rate


tool_kit = model.bind_tools([
    get_currency_factor,
    convertCurrency
])


query = HumanMessage(content="What is USD in INR?")
message = [query]


# AI Message
ai_msg = tool_kit.invoke(message)
message.append(ai_msg)


for tool_call in ai_msg.tool_calls:

    # Execute the 1st tool and get the conversion rate
    if tool_call['name'] == 'get_currency_factor':

        tool_result = get_currency_factor.invoke(tool_call['args'])

        # Store conversion rate
        conversion_rate = tool_result

        # Append tool result to messages
        message.append(ToolMessage(content=str(tool_result),tool_call_id=tool_call['id']))


    # Execute the 2nd tool using conversion rate
    if tool_call['name'] == 'convertCurrency':

        #add to 2 tool call
        tool_call['args']['conversion_rate'] = conversion_rate

        tool_result = convertCurrency.invoke(tool_call['args'])

        # Append tool result to messages
        message.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call['id']
            )
        )


# Ask LLM for final answer
conversion_rate = get_currency_factor.invoke({
    "base_currency": "USD",
    "target_currency": "INR"
})

result = convertCurrency.invoke({
    "base_currency_value": 400,
    "conversion_rate": conversion_rate
})

print(result)
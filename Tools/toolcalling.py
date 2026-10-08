from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

#binding
llm_with_tools = llm.bind_tools([multiply])

#Human message
query = HumanMessage('can you multiply 3 with 1000')
messages = [query]

#AImessage
result = llm_with_tools.invoke(messages)
messages.append(result)

#Tool message
tool_result = multiply.invoke(result.tool_calls[0])

messages.append(tool_result)

#All message to LLM
result=llm_with_tools.invoke(messages).content
print(result)
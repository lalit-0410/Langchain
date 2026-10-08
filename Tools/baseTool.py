from langchain_community.tools import StructuredTool,BaseTool
from pydantic import BaseModel,Field
from typing import Type

class MultiplyInput(BaseModel):
    a:int=Field(required=True, description="First number to multiply")
    b:int=Field(required=True, description="Second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

tool=MultiplyTool().invoke({'a':14,'b':5})
print(tool)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#load api key
load_dotenv()

llm=ChatOpenAI(model="gpt-6-luna")
result=llm.invoke("What is capital of india?")
print(result.content)
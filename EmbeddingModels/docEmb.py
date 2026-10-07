from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 

embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = [
    "Python is a programming language.",
    "Java is used for backend development.",
    "Spring Boot is a Java framework.",
    "LangChain is used to build LLM applications.",
    "Qdrant is a vector database."
]
#for single
result=embedding.embed_documents(documents)

print(str(result))
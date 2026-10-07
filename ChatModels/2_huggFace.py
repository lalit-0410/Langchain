from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


# Load .env first
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="ibm-granite/granite-4.2-30b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Uttar Pradesh?")

print(result.content)
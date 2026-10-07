from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatOpenAI(model='gpt-4')

prompt=PromptTemplate(
    template="Give 2 interesting fact about {topic}",
    input_variables=['topic']


)

parcer=StrOutputParser()

chain=prompt | model | parcer
result=chain.invoke({'topic':'basketball'})
print(result)

chain.get_graph().print_ascii()
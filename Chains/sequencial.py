from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1=PromptTemplate(
    template='Generate a report about {topic} about 100 words ',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarize the report in 2 points {text} ',
    input_variables=['text']
)

model=ChatOpenAI(model='gpt-4')

parcer=StrOutputParser()

chain=prompt1 | model | parcer | prompt2 | model | parcer
result=chain.invoke({'topic':'phone'}) 
print(result)
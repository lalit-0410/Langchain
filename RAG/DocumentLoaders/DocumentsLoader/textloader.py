from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

prompt=PromptTemplate(
    template='Summarize in simple points {text}',
    input_variables=['text']
)

model=ChatOpenAI(model='gpt-4')

parser=StrOutputParser()


loader=TextLoader('cricket.txt',encoding='utf-8')
doc=loader.load()

chain=prompt | model | parser 

print(chain.invoke({'text':doc[0].page_content}))
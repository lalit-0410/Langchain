from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatOpenAI(model='gpt-4')

prompt=PromptTemplate(
    template='Generate a tweet on {topic} for twitter\n',
    input_variables=['topic']
)

parcer=StrOutputParser()

prompt2=PromptTemplate(
    template='Generate a post on {topic} for linkedin\n',
    input_variables=['topic']
)

parallel_chain=RunnableParallel(
    tweet=RunnableSequence(prompt,model,parcer),
    linkedin=RunnableSequence(prompt2,model,parcer)
)
result=parallel_chain.invoke({'topic':'AI'})
print(result['tweet'])
print(result['linkedin'])
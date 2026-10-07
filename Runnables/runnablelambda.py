from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnableSequence,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatOpenAI(model='gpt-4')

prompt=PromptTemplate(
    template='Generate a joke on {topic}\n',
    input_variables=['topic']
)

parcer=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt,model,parcer)

parallel_joke_process=RunnableParallel(
    original=RunnablePassthrough(),
    word_count=RunnableLambda(lambda x:len(x.split()))
)

final_chain=RunnableSequence(joke_gen_chain,parallel_joke_process)
result=final_chain.invoke({'topic':'clouds'})
print(result['original'])
print(result['word_count'])
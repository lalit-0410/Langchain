from langchain_core.runnables import RunnableSequence,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatOpenAI(model='gpt-4')

prompt=PromptTemplate(
    template='Generate a joke on {topic}',
    input_variables=['topic']
)

parcer=StrOutputParser()

prompt2=PromptTemplate(
    template='Explain the joke {joke}',
    input_variables=['joke']
)
joke_chain=RunnableSequence(prompt,model,parcer)

sequence_chain = (
    joke_chain
    | RunnableLambda(lambda joke: {"joke": joke})
    | prompt2
    | model
    | parcer
)

print(joke_chain.invoke({'topic':'Cricket'}))
print(sequence_chain.invoke({"topic": "Cricket"}))


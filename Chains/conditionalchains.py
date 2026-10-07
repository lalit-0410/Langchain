from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers  import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal


load_dotenv()

model=ChatOpenAI(model='gpt-4')
parcer=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Classify the given feedback is positive or negative')

parcer2=PydanticOutputParser(pydantic_object=Feedback)



prompt=PromptTemplate(
    template='Analyze the {feedback} is positive or negative \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parcer2.get_format_instructions()}
)

classifier_chain=prompt | model | parcer2

prompt2=PromptTemplate(
    template='Write a appropriate response to this  positive feedback\n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write a appropriate response to this  negative feedback\n {feedback}',
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2 | model | parcer),
    (lambda x:x.sentiment=='negative', prompt3 | model | parcer),
    RunnableLambda(lambda x:"Could not find sentiment")
    )

chain=classifier_chain | branch_chain


result=chain.invoke({'feedback':'This device is not good'})
print(result)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers  import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence

load_dotenv()


prompt1=PromptTemplate(
    template="Generate short notes on given {text} ",
    input_variables=['text']

)


prompt2=PromptTemplate(
    template="Generate 5 quiz questions on given {text} ",
    input_variables=['text']
    
)

prompt3=PromptTemplate(
    template="Merge the following notes and quiz into a single document\n notes-> {notes}\n and quiz-> {quiz}",
    input_variables=['notes','quiz']
    
)

model=ChatOpenAI(model='gpt-4')


parcer=StrOutputParser()

parallel_chain = RunnableParallel(
    notes=prompt1 | model | parcer,
    quiz=prompt2 | model | parcer
)
merge_chain=prompt3 | model | parcer

chain = RunnableSequence(parallel_chain , merge_chain)

text="""
A vector database is a type of database built to store and search special kinds of data called vector embeddings. These embeddings are numbers that represent the meaning or characteristics of things such as text, images, video, or audio.
While traditional databases work best with neatly organised data in rows and columns, vector databases are designed for working with unstructured, multi‑dimensional data. Their main job is to quickly find things that are similar to each other—known as similarity search—even if they aren’t exact matches, by comparing how close their embeddings are in mathematical space.
This makes vector databases especially useful for modern artificial intelligence (AI) applications. They power semantic search, which returns results based on meaning rather than exact words, and they support generative AI tools by helping pull in the most relevant information when creating answers, images, or other content.
Vector databases are also used in recommendation engines, image and video search, and language comprehension. In short, they make it possible for AI systems to search and match information in a way that is much closer to how humans think and understand.
"""

result=chain.invoke({'text':text})
print(result)
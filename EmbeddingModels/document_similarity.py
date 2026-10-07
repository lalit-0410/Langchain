from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

#ml
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=300)
documents = [
    "Virat Kohli is an Indian cricketer known for his batting and leadership.",
    "Rohit Sharma is an Indian cricketer and an excellent opening batsman.",
    "MS Dhoni is a former Indian captain known for his wicketkeeping and finishing skills.",
    "Jasprit Bumrah is an Indian fast bowler known for his accuracy and unique bowling action.",
    "Sachin Tendulkar is a legendary Indian batsman and one of the greatest cricketers of all time."
]
query="Tell me about ms dhoni"

doc_embedding=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

#always remember to pass list in 2d
#print(cosine_similarity([query_embedding],doc_embedding))


#jo list return ho rahi 300D wali usme pehla [0]
scores=cosine_similarity([query_embedding],doc_embedding)[0]


#This adds the index to every score:
#print(list(enumerate(scores)))

#highest score fetch
highest = np.argmax(scores)
score = scores[highest]

print(query)

#Text return
print(documents[highest])

print("similarity score is ",score)


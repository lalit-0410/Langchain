from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda

chain = RunnablePassthrough()

print(chain.invoke("Hello"))
chain2=RunnableParallel(
    original=RunnablePassthrough(),
    upper=RunnableLambda(lambda x: x.upper())
)
print(chain2.invoke("Hello"))
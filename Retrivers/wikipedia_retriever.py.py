from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    doc_content_chars_max=4000
)

docs = retriever.invoke("Virat Kohli")

for doc in docs:
    print("CONTENT:")
    print(doc.page_content)

    print("\nMETADATA:")
    print(doc.metadata)

    print("=" * 80)
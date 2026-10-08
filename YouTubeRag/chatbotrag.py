import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥"
)

st.title("🎥 YouTube RAG Chatbot")
st.write("Ask questions about the YouTube video.")

video_id = st.text_input(
    "Enter YouTube Video ID",
    placeholder="Example: Gfr50f6ZBvo"
)

process_button = st.button("Process Video")


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

#Step 1 -> Loading
if process_button:

    if not video_id:
        st.warning("Please enter a YouTube Video ID.")
        st.stop()

    try:

        with st.spinner("Fetching YouTube transcript..."):

            ytt_api = YouTubeTranscriptApi()

            transcript_data = ytt_api.fetch(
                video_id,
                languages=["en"]
            )

            transcript = " ".join(
                snippet.text
                for snippet in transcript_data
            )

        st.success("Transcript fetched successfully!")

        # --------------------------------------------------
        # 2. TEXT SPLITTING
        # --------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.create_documents([transcript])

        # --------------------------------------------------
        # 3. EMBEDDINGS
        # --------------------------------------------------

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=400
        )


        # --------------------------------------------------
        # 4. VECTOR STORE
        # --------------------------------------------------

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )


        # --------------------------------------------------
        # 5. RETRIEVER
        # --------------------------------------------------

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )


        # --------------------------------------------------
        # 6. MODEL
        # --------------------------------------------------

        model = ChatOpenAI(
            model="gpt-4",
            temperature=0.2
        )


        # --------------------------------------------------
        # 7. PROMPT
        # --------------------------------------------------

        prompt = PromptTemplate(
            template="""
You are a helpful YouTube video assistant.

Answer the user's question ONLY using the provided
video transcript context.

If the answer is not present in the context,
say:

"I don't know based on the video."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
""",
            input_variables=["context", "question"]
        )


        # --------------------------------------------------
        # FORMAT DOCUMENTS
        # --------------------------------------------------

        def format_docs(docs):

            return "\n\n".join(
                doc.page_content
                for doc in docs
            )


        # --------------------------------------------------
        # RAG CHAIN
        # --------------------------------------------------

        parallel_chain = RunnableParallel({

            "context": retriever
                | RunnableLambda(format_docs),

            "question": RunnablePassthrough()

        })


        main_chain = (
            parallel_chain
            | prompt
            | model
            | StrOutputParser()
        )


        # Store chain in session
        st.session_state["chain"] = main_chain

        st.session_state["video_processed"] = True

        st.success("✅ Video is ready! You can now ask questions.")

    except TranscriptsDisabled:

        st.error(
            "This video does not have captions/transcripts available."
        )

    except Exception as e:

        st.error(f"Something went wrong: {e}")


# --------------------------------------------------
# QUESTION ANSWERING
# --------------------------------------------------

if st.session_state.get("video_processed", False):

    st.subheader("💬 Ask Questions About the Video")

    question = st.text_input(
        "Your Question",
        placeholder="Example: What is discussed about nuclear fusion?"
    )

    ask_button = st.button("Ask Question")

    if ask_button:

        if not question:

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching video and generating answer..."):

                answer = st.session_state["chain"].invoke(
                    question
                )

            st.subheader("🤖 Answer")

            st.write(answer)
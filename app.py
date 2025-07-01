import os
from dotenv import load_dotenv
import streamlit as st
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import requests
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="YouTube RAG", page_icon="📽️")
st.title("🎥 YouTube RAG with Groq + Streamlit")

# Inputs
video_id = st.text_input("Enter YouTube Video ID (e.g. pRpeEdMmmQ0):")
question = st.text_input("Ask your question based on the video (e.g. 'What is the Hindi meaning of this song?' or 'What is the title of this video?')")

if st.button("Get Answer"):
    if not video_id.strip():
        st.warning("⚠️ Please enter a valid YouTube Video ID.")
        st.stop()
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
        st.stop()

    with st.spinner("🔍 Fetching transcript and generating answer..."):
        # Try getting transcript (English first, fallback to Hindi)
        transcript_lang = "en"
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except NoTranscriptFound:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
                transcript_lang = "hi"
                # st.info("ℹ️ English transcript not found. Using Hindi transcript instead.")
            except NoTranscriptFound:
                # st.error("❌ No transcript found in English or Hindi.")
                st.stop()

        # Join transcript
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        st.caption(f"📄 Transcript language used: `{transcript_lang.upper()}`")

        # Try to get video title using pytube
        # try:
        #     print(video_id)
        #     yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        #     print(yt)
        #     video_title = yt.title
        # except Exception as e:
        #     video_title = "Unknown Title"
        #     st.warning(f"⚠️ Could not fetch video title: {e}")
        # --- Video Title Fetching ---
        try:
            # Try using pytube first
            from pytube import YouTube
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            video_title = yt.title
        except Exception as e:
            # If pytube fails, fallback to oEmbed API
            try:
                oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
                response = requests.get(oembed_url)
                if response.status_code == 200:
                    video_title = response.json().get("title", "Unknown Title")
                else:
                    video_title = "Unknown Title"
                # st.warning(f"⚠️ pytube failed. Used fallback oEmbed (status: {response.status_code})")
            except Exception as oe:
                video_title = "Unknown Title"
                # st.warning(f"⚠️ Could not fetch video title using any method: {oe}")
        # Text splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(transcript)

        # Embedding model
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

        # Vector store and retriever
        vector_store = FAISS.from_texts(chunks, embedding)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        retrieved_docs = retriever.invoke(question)

        # Combine video title with retrieved context
        context_text = f"Video Title: {video_title}\n\n" + "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Custom instruction logic
        if "hindi meaning" in question.lower():
            custom_instruction = "Translate the meaning of the following song into Hindi."
        elif "title" in question.lower():
            custom_instruction = "If the question is about the title, use the video title provided in the context below."
        else:
            custom_instruction = (
                "Use the pieces of information provided in the context to answer the user's question.\n"
                "If you don't know the answer, just say that you don't. Don't try to make up an answer.\n"
                "Don't provide anything out of the given context."
            )

        # Final prompt template
        CUSTOM_PROMPT_TEMPLATE = """
{instruction}

Context:
{context}

Question: {question}

Start the answer directly, no small talk please.
"""
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["instruction", "context", "question"]
        )

        final_prompt = prompt.invoke({
            "instruction": custom_instruction,
            "context": context_text,
            "question": question
        })

        # Generate answer with Groq
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
        answer = llm.invoke(final_prompt)

        # Show result
        st.subheader("📘 Answer:")
        st.success(answer.content)

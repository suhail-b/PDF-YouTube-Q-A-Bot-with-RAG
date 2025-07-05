import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from youtube_to_text import youtube_to_text

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

st.title("RAG Bot")
mode = st.radio("Select Input Type", ["YouTube", "PDF"])

def reset_state():
    for key in ["docs", "rag"]:
        if key in st.session_state:
            del st.session_state[key]

if mode == "YouTube":
    url = st.text_input("YouTube URL")
    if st.button("Process YouTube"):
        reset_state()  # 👈 Reset before creating new embeddings
        try:
            text = youtube_to_text(url)
            st.session_state.docs = [Document(page_content=text)]
            st.success("Transcription done")
        except Exception as e:
            st.error(f"Error: {e}")

elif mode == "PDF":
    file = st.file_uploader("Upload PDF", type="pdf")
    if file and st.button("Process PDF"):
        reset_state()  # 👈 Reset before creating new embeddings
        with open(file.name, "wb") as f: f.write(file.getvalue())
        loader = PyPDFLoader(file.name)
        st.session_state.docs = loader.load()
        st.success("PDF loaded")

if "docs" in st.session_state and "rag" not in st.session_state:
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(st.session_state.docs)
    db = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    retriever = db.as_retriever()
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, api_key=groq_key)
    st.session_state.rag = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

if "rag" in st.session_state:
    q = st.text_input("Ask a question")
    if q:
        a = st.session_state.rag.invoke(q)
        st.write("### 🤖 Answer")
        st.write(a)

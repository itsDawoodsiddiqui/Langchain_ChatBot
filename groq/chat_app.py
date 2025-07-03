import streamlit as st
import os
import time
from dotenv import load_dotenv

# Embeddings, loaders, chains, etc.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("🚨 Please set GROQ_API_KEY in your .env file.")
    st.stop()

# ——————————————————————————————
# 1) Build or reuse your FAISS index once
# ——————————————————————————————
if "vector" not in st.session_state:
    # a) Choose an HF embedder
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # b) Load your docs
    loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
    docs = loader.load()

    # c) Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # d) Build FAISS
    st.session_state.vectors = FAISS.from_documents(
        chunks, st.session_state.embeddings
    )
    st.session_state.vector = True

# ——————————————————————————————
# 2) UI + LLM setup
# ——————————————————————————————
st.title("ChatGroq Demo")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="deepseek-r1-distill-llama-70b",
)

# ——————————————————————————————
# 3) Prompt & chain
# ——————————————————————————————
# Note: using {context} here, so document_variable_name must be "context"
prompt = ChatPromptTemplate.from_template(
    """Use the following context to answer the question:
{context}

Question: {input}
"""
)

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="context",
)

retriever       = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ——————————————————————————————
# 4) Interaction
# ——————————————————————————————
user_input = st.text_input("Input your prompt here")
if user_input:
    start  = time.process_time()
    result = retrieval_chain.invoke({"input": user_input})
    elapsed = round(time.process_time() - start, 2)

    st.write(f"⏱️ Response time: {elapsed}s")
    st.write(result["answer"])

    with st.expander("📄 Document similarity search"):
        for doc in result["context"]:
            st.write(doc.page_content)
            st.write("---")
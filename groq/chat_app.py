import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("🚨 Please set GROQ_API_KEY in your .env file.")
    st.stop()

st.title("ChatGroq Demo")

uploaded_file = st.file_uploader("📄 Upload your PDF file", type="pdf")

if uploaded_file:
    with st.spinner("📚 Processing PDF..."):
        # Save uploaded file to disk
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Create Embeddings and Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        # Setup Groq LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="deepseek-r1-distill-llama-70b"
        )

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
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # User input
        query = st.text_input("💬 Ask a question based on the uploaded PDF")
        if query:
            start = time.process_time()
            result = retrieval_chain.invoke({"input": query})
            elapsed = round(time.process_time() - start, 2)

            st.write(f"⏱️ Response time: {elapsed}s")
            st.write(result["answer"])

            with st.expander("🔍 Relevant Document Chunks"):
                for doc in result["context"]:
                    st.write(doc.page_content)
                    st.write("---")
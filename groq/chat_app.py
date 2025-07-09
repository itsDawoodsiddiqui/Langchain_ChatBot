import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("🚨 Please set GROQ_API_KEY in your .env file.")
    st.stop()

# App setup
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with PDF using Groq + ChromaDB")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Sidebar with chat history
with st.sidebar:
    st.subheader("🕘 Chat History")
    if st.session_state.chat_history:
        for i, item in enumerate(st.session_state.chat_history[::-1], 1):
            st.markdown(f"{i}. **{item['query']}**")
    else:
        st.write("No queries yet.")
    
    if st.button("🧹 Clear Memory"):
        st.session_state.chat_history = []
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        st.success("Memory cleared!")

# File uploader
uploaded_file = st.file_uploader("📄 Upload your PDF file", type="pdf")

if uploaded_file:
    with st.spinner("📚 Processing PDF..."):
        # Save file
        pdf_path = "temp_files/temp.pdf"
        os.makedirs("temp_files", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load and split
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Embeddings + ChromaDB
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chroma_dir = "new_folder/chroma_db"
        os.makedirs(chroma_dir, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir
        )
        vectorstore.persist()

        # RAG pipeline
        retriever = vectorstore.as_retriever()
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")
        prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question:
{context}

Question: {input}
""")
        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Save to session
        st.session_state.vectorstore_ready = True
        st.session_state.retrieval_chain = retrieval_chain

# User question (always visible)
query = st.text_input("💬 Ask a question (PDF optional)")

if query:
    start = time.process_time()
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")

    if st.session_state.vectorstore_ready and st.session_state.retrieval_chain:
        # Use RAG from PDF
        result = st.session_state.retrieval_chain.invoke({"input": query})
        answer = result["answer"]
        print("result", result)        
        print("answer", answer)

        st.write(f"⏱️ Response time: {round(time.process_time() - start, 2)}s")
        st.write(answer)

        with st.expander("🔍 Relevant Document Chunks"):
            for doc in result["context"]:
                st.write(doc.page_content)
                st.markdown("---")
    else:
        # Use conversational memory
        st.session_state.messages.append({"role": "user", "content": query})
        response = llm.invoke(st.session_state.messages)
        answer = response.content
        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.write(f"⏱️ Response time: {round(time.process_time() - start, 2)}s")
        st.write(answer)

    # Save to chat history
    st.session_state.chat_history.append({
        "query": query,
        "answer": answer
    })



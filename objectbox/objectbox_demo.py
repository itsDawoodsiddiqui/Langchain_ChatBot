import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Setup Streamlit UI
st.title("Objectbox VectorstoreDB With Llama3 Demo")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    {context}
    Question: {input}
    """
)

# Function to embed documents
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('./us_census')  # Make sure this path exists
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )

        # Ensure you have correct ObjectBox version that supports this
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=68  # You can try removing this if error persists
        )



# Input field for user question
input_prompt = st.text_input("Enter Your Question From Documents")

# UI – Button to embed documents
if st.button("Documents Embedding"):
    try:
        vector_embedding()
        st.success("✅ ObjectBox Database is ready.")
    except Exception as e:
        st.error(f"❌ Embedding failed: {e}")
        
        
# If input given, run retrieval
if input_prompt:
    if "vectors" not in st.session_state:
        st.warning("📄 Please click 'Documents Embedding' first to load your document data.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': input_prompt})
        st.write("🧠 Answer:")
        st.write(response['answer'])

        with st.expander("📄 Document Chunks Retrieved"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.markdown("---")

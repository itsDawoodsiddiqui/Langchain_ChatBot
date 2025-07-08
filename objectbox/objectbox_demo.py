import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import whisper
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path

# Init LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

{context}
Question: {input}
""")

# Setup Streamlit UI
st.title("Multimodal GPT Chatbot: Text + Image + Audio")

# OCR function
def process_image(image_path):
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

# Audio transcription function
def transcribe_audio(audio_path):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")
        
        model = whisper.load_model("base")  # or "tiny" for faster
        
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""


# Embedding vectorstore
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('./uscensus')
        st.session_state.docs = st.session_state.loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=68
        )

# Upload section
uploaded_file = st.file_uploader("Upload PDF, Image, or Audio", type=["pdf", "jpg", "jpeg", "png", "mp3", "wav", "m4a"])
file_text = ""

if uploaded_file:
    os.makedirs("temp_files", exist_ok=True)
    file_ext = uploaded_file.name.split('.')[-1].lower()
    saved_path = os.path.join("temp_files", uploaded_file.name)

    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if not os.path.exists(saved_path):
        st.error("❌ File save failed. Try again.")
    else:
        st.success(f"📁 File Uploaded: {uploaded_file.name}")

        if file_ext in ["jpg", "jpeg", "png"]:
            st.image(saved_path, caption="Uploaded Image")
            file_text = process_image(saved_path)
            st.session_state.context_text = file_text
            st.success("✅ Text extracted from image.")

        elif file_ext in ["mp3", "wav", "m4a"]:
            file_text = transcribe_audio(saved_path)
            st.session_state.context_text = file_text
            st.audio(uploaded_file)
            st.success("✅ Audio transcribed successfully.")

        elif file_ext == "pdf":
            st.write("ℹ️ Click 'Documents Embedding' to load PDF into ObjectBox.")

# Input field
input_prompt = st.text_input("Ask a question based on uploaded content")

if input_prompt:
    if "context_text" in st.session_state:
        # For audio/image
        context = [Document(page_content=st.session_state.context_text)]
        document_chain = create_stuff_documents_chain(llm, prompt)
        response = document_chain.invoke({"input": input_prompt, "context": context})
        st.write("🤖 Answer:")
        st.write(response)

    elif "vectors" in st.session_state:
        # For PDFs
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': input_prompt})
        st.write("🤖 Answer:")
        st.write(response['answer'])
        with st.expander("📄 Document Chunks Retrieved"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.markdown("---")
    else:
        st.warning("📥 Please upload a file or embed documents first.")

# Embed PDF button
if st.button("Documents Embedding"):
    try:
        vector_embedding()
        st.success("✅ ObjectBox Vectorstore is ready.")
    except Exception as e:
        st.error(f"❌ Embedding failed: {e}")

# Imports
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")

# Set environment (for LangChain tracing/debugging if needed)
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Dark mode styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0e0e0e;
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #1e1e1e;
            color: white;
            border-radius: 6px;
            padding: 10px;
            font-size: 16px;
        }
        h1, h2, h3, h4, h5, h6, label {
            color: white !important;
        }
        .stMarkdown {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# UI Title
st.title("🧠 LangChain + GROQ Chatbot")
input_text = st.text_input("Ask a question:")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# LLM & Chain
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    api_key=groq_api_key
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Output
if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": input_text})
        st.success("✅ Response:")
        st.markdown(response)


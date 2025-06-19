#  Imports 
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

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# UI 
st.title('🧠 LangChain + GROQ Chatbot')
input_text = st.text_input("Ask a question:")

#  LLM & Chain 
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0,
    api_key=groq_api_key
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Output 
if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({'question': input_text})
        st.success("✅ Response:")
        st.write(response)

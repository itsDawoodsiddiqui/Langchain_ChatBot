from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from pathlib import Path

# --- Load .env from parent directory ---
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- Safely read GROQ_API_KEY ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file.")
os.environ["GROQ_API_KEY"] = groq_api_key

# --- FastAPI App ---
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# --- Define ChatGroq model once ---
model = ChatGroq(
    model="gemma2-9b-it",
    api_key=groq_api_key
)

# --- Optional Ollama model ---
llm = Ollama(model="tinyllama:1.1b")

# --- Prompt templates ---
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5-year-old child with 100 words")

# --- Routes (use `model`, not duplicate ChatGroq)
add_routes(app, model, path="/openai")
add_routes(app, prompt1 | model, path="/essay")
add_routes(app, prompt2 | llm, path="/poem")

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

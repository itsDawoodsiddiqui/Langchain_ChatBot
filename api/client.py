import requests
import streamlit as st

# --- Get response from ChatGroq model (Essay) ---
def get_groq_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={'input': {'topic': input_text}}
    )
    print(response)
    return response.json()['output']['content']

# --- Get response from Ollama model (Poem) ---
def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={"input": {"topic": input_text}}
    )
    # 1️⃣ Debug output:
    print(f"[poem/invoke] status: {response.status_code}")
    print(f"[poem/invoke] body: {response.text!r}")

    # 2️⃣ Guard against non-200
    if response.status_code != 200:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")

    # 3️⃣ Now it’s safe to decode
    payload = response.json()
    return payload.get("output")

# --- Streamlit UI ---
st.title('LangChain Demo with ChatGroq + Ollama')

input_text = st.text_input("📝 Write an essay on:")
input_text1 = st.text_input("🪶 Write a poem on:")

if input_text:
    with st.spinner("Generating essay from ChatGroq..."):
        st.write(get_groq_response(input_text))

if input_text1:
    with st.spinner("Generating poem from LLaMA2..."):
        st.write(get_ollama_response(input_text1))

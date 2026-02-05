# Imports
import json
import requests
import chromadb
import gradio as gr
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
import os

load_dotenv(override=True)

# Configuration
CHROMA_PATH = r"C:\Users\mah19\RAG\chroma_db"
COLLECTION_NAME = "growing_vegetables"
TOP_K = 4

# Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"

# OpenAI
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# HuggingFace
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_API_TOKEN = os.getenv("HF_TOKEN")


# Init ChromaDB
def init_chroma_collection(path: str, collection_name: str):
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(name=collection_name)


# Retrieve docs
def retrieve_documents(collection, query: str, top_k: int) -> List[str]:
    results = collection.query(query_texts=[query], n_results=top_k)
    return results.get("documents", [[]])[0]


# Build prompt
def build_prompt(context_docs: List[str], user_query: str) -> str:
    context = "\n\n".join(context_docs)
    return f"""
You are a helpful assistant.
You answer questions about growing vegetables in Florida.
Only answer using the data below.
If you don't know, say: I don't know.

--------------------
Data:
{context}

Question: {user_query}
"""


# Ollama call
def call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    full = ""
    for line in response.iter_lines():
        if line:
            decoded = json.loads(line)
            full += decoded.get("response", "")
    return full.strip()


# OpenAI call
def call_openai(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    res = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()


# HuggingFace call
def call_huggingface(prompt: str) -> str:
        client = InferenceClient(api_key=HF_API_TOKEN,)    
        res = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
        return res.choices[0].message.content.strip()


# Router
def generate_answer(prompt: str, provider: str) -> str:
    if provider == "Ollama (Local)":
        return call_ollama(prompt)
    if provider == "OpenAI":
        return call_openai(prompt)
    if provider == "HuggingFace":
        return call_huggingface(prompt)
    return "Invalid model"


# Main logic
def answer_question(user_query: str, provider: str) -> str:
    collection = init_chroma_collection(CHROMA_PATH, COLLECTION_NAME)
    docs = retrieve_documents(collection, user_query, TOP_K)
    if not docs:
        return "I don't know"
    prompt = build_prompt(docs, user_query)
    return generate_answer(prompt, provider)


# Gradio UI
with gr.Blocks(title="RAG – Growing Vegetables (Florida)") as demo:
    gr.Markdown("# 🌱 RAG Assistant (Multi-LLM)")

    provider = gr.Dropdown(
        ["Ollama (Local)", "OpenAI", "HuggingFace"],
        value="Ollama (Local)",
        label="LLM Provider"
    )

    query = gr.Textbox(label="Your Question")
    output = gr.Textbox(label="Answer", lines=8)

    gr.Button("Ask").click(
        fn=answer_question,
        inputs=[query, provider],
        outputs=output
    )


# Entry point
if __name__ == "__main__":
    demo.launch()

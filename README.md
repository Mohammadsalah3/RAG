# 🌱 RAG Assistant – Multi-LLM (Ollama, OpenAI, Hugging Face)

A **Retrieval-Augmented Generation (RAG)** application that answers questions about **growing vegetables in Florida** using a vector database (**ChromaDB**) and multiple LLM providers.

The project supports:
- 🖥️ Local LLMs via **Ollama**
- ☁️ Cloud LLMs via **OpenAI**
- 🤗 Open-source models via **Hugging Face**

A simple **Gradio UI** allows switching between models at runtime.

---

## 🚀 Features

- Retrieval-Augmented Generation (RAG)
- ChromaDB persistent vector store
- Multi-LLM routing (Local + Cloud)
- Anti-hallucination prompt design
- Gradio web interface
- Modular and extensible codebase

---

## 🧠 RAG Pipeline
- User Query
- ChromaDB (Top-K Retrieval)
- Context Construction
- LLM (Ollama / OpenAI / Hugging Face)
- Final Answer
---

## 🛠️ Tech Stack

- **Python**
- **ChromaDB** – Vector database
- **Ollama** – Local LLM inference
- **OpenAI API**
- **Hugging Face Inference API**
- **Gradio** – UI

---
title: Advanced RAG Backend
emoji: 🤖
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8000
---

# Advanced AI PDF RAG Chatbot Backend

This is the FastAPI backend for the Advanced PDF RAG Chatbot. It runs in a Docker container on Hugging Face Spaces and uses persistent storage to store the document registry and ChromaDB vector store.

## Local Development

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

# Local RAG Knowledge Assistant

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using local LLMs.

## Features
- Document loading using LangChain
- Text chunking
- Embeddings using Ollama (Llama3)
- FAISS vector store for similarity search
- Context-aware question answering

## Tech Stack
- LangChain
- Ollama (Llama3)
- FAISS

## How it works
1. Load text file
2. Split into chunks
3. Generate embeddings
4. Store in FAISS
5. Retrieve relevant chunks
6. Generate answer using LLM

## Run
```bash
python src/rag_faiss_ollama.py

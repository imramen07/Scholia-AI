# Scholia-AI

A streamlit-based PDF QnA chatbot using langchain, FAISS, HuggingFace embeddings and OLLAMA LLM
It allows users to upload PDFs and ask questions with GPU CUDA acceleration support.
Optimized using MMR, intent detection and clean UI.

## Features
- Multiple PDF upload and indexing
- GPU enabled embeddings and reranking
- LLM driven QnA with context extraction
- Chat history support

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py

requires ollama\llama3 installed and running
# Research Paper Answer Bot - RAG System
# Capstone Project for Analytics Vidya

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system for answering questions about AI and Data Science research papers. The system can index research papers, retrieve relevant context, and generate accurate responses with source attribution.

## Features
- Multiple embedding model support (HuggingFace and OpenAI)
- Various retrieval strategies (cosine similarity, hybrid search, rerankers)
- Vector database indexing with ChromaDB and FAISS
- Source document tracking and attribution
- Streamlit web interface
- Multi-user conversational support

## Installation
1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env` file
3. Run the application: `streamlit run app.py`

## Project Structure
```
├── data/                   # Research papers and processed documents
├── embeddings/            # Embedding models and vector stores
├── retrieval/             # Different retrieval strategies
├── rag/                   # Core RAG pipeline components
├── utils/                 # Utility functions
├── tests/                 # Test cases and evaluation
├── app.py                 # Streamlit application
├── config.py              # Configuration settings
└── requirements.txt       # Dependencies
```

## Usage
1. Load research papers into the system
2. Choose embedding model and retrieval strategy
3. Ask questions about AI/ML topics
4. Get answers with source document references

## Evaluation Metrics
- Response relevance and accuracy
- Source attribution quality
- Retrieval performance
- User satisfaction scores

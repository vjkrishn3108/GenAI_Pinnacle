# Configuration settings for the RAG system

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Model configurations
EMBEDDING_MODELS = {
    "openai": {
        "model_name": "text-embedding-3-small",
        "dimensions": 1536
    },
    "huggingface": {
        "sentence_transformer": "all-MiniLM-L6-v2",
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
        "domain_specific": "allenai-specter"
    }
}

# LLM configurations
LLM_CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.1,
    "max_tokens": 1000
}

# Vector database settings
VECTOR_DB_CONFIG = {
    "chroma": {
        "persist_directory": "./embeddings/chroma_db",
        "collection_name": "research_papers"
    },
    "faiss": {
        "index_path": "./embeddings/faiss_index"
    }
}

# Retrieval settings
RETRIEVAL_CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.7,
    "rerank_top_k": 3
}

# Document processing settings
DOCUMENT_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "supported_formats": [".pdf", ".txt", ".docx"]
}

# Data paths
DATA_PATHS = {
    "raw_documents": "./data/raw",
    "processed_documents": "./data/processed",
    "embeddings": "./embeddings",
    "logs": "./logs"
}

# Create directories if they don't exist
for path in DATA_PATHS.values():
    os.makedirs(path, exist_ok=True)

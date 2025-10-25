"""
Embedding models module for different embedding strategies
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Base class for embedding models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        raise NotImplementedError
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        raise NotImplementedError
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        raise NotImplementedError


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model wrapper"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        super().__init__(model_name)
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        self.model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=config.OPENAI_API_KEY
        )
        self.dimension = config.EMBEDDING_MODELS["openai"]["dimensions"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI"""
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents with OpenAI: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using OpenAI"""
        try:
            return self.model.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query with OpenAI: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """HuggingFace embedding model wrapper"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using HuggingFace"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents with HuggingFace: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using HuggingFace"""
        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query with HuggingFace: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class EmbeddingFactory:
    """Factory class for creating embedding models"""
    
    @staticmethod
    def create_embedding_model(model_type: str, model_name: Optional[str] = None) -> EmbeddingModel:
        """Create an embedding model based on type"""
        
        if model_type.lower() == "openai":
            model_name = model_name or config.EMBEDDING_MODELS["openai"]["model_name"]
            return OpenAIEmbeddingModel(model_name)
        
        elif model_type.lower() == "huggingface":
            model_name = model_name or config.EMBEDDING_MODELS["huggingface"]["sentence_transformer"]
            return HuggingFaceEmbeddingModel(model_name)
        
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get list of available embedding models"""
        return {
            "openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002"
            ],
            "huggingface": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2",
                "allenai-specter",
                "sentence-transformers/all-MiniLM-L12-v2"
            ]
        }


class EmbeddingEvaluator:
    """Evaluate different embedding models"""
    
    def __init__(self):
        self.test_queries = [
            "What is transformer architecture?",
            "How does attention mechanism work?",
            "What are the applications of large language models?",
            "Explain the concept of fine-tuning in machine learning",
            "What is the difference between supervised and unsupervised learning?"
        ]
    
    def evaluate_model(self, model: EmbeddingModel, documents: List[str]) -> Dict[str, Any]:
        """Evaluate an embedding model"""
        results = {
            "model_name": model.model_name,
            "dimension": model.get_embedding_dimension(),
            "embedding_time": 0,
            "query_embedding_time": 0,
            "similarity_scores": []
        }
        
        import time
        
        # Test document embedding
        start_time = time.time()
        doc_embeddings = model.embed_documents(documents[:10])  # Test with first 10 docs
        results["embedding_time"] = time.time() - start_time
        
        # Test query embedding
        start_time = time.time()
        query_embeddings = [model.embed_query(query) for query in self.test_queries]
        results["query_embedding_time"] = time.time() - start_time
        
        # Calculate similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        
        for i, query_emb in enumerate(query_embeddings):
            similarities = cosine_similarity([query_emb], doc_embeddings)[0]
            results["similarity_scores"].append({
                "query": self.test_queries[i],
                "max_similarity": float(np.max(similarities)),
                "mean_similarity": float(np.mean(similarities))
            })
        
        return results
    
    def compare_models(self, models: List[EmbeddingModel], documents: List[str]) -> Dict[str, Any]:
        """Compare multiple embedding models"""
        results = {}
        
        for model in models:
            results[model.model_name] = self.evaluate_model(model, documents)
        
        return results


if __name__ == "__main__":
    # Test embedding models
    factory = EmbeddingFactory()
    
    # Test HuggingFace model (no API key required)
    try:
        hf_model = factory.create_embedding_model("huggingface")
        print(f"HuggingFace model loaded: {hf_model.model_name}")
        print(f"Embedding dimension: {hf_model.get_embedding_dimension()}")
        
        # Test embedding
        test_texts = ["This is a test document.", "Another test document."]
        embeddings = hf_model.embed_documents(test_texts)
        print(f"Generated {len(embeddings)} embeddings")
        
    except Exception as e:
        print(f"Error testing HuggingFace model: {e}")
    
    # Test OpenAI model (requires API key)
    try:
        openai_model = factory.create_embedding_model("openai")
        print(f"OpenAI model loaded: {openai_model.model_name}")
        print(f"Embedding dimension: {openai_model.get_embedding_dimension()}")
        
    except Exception as e:
        print(f"Error testing OpenAI model: {e}")
        print("Make sure to set OPENAI_API_KEY in .env file")

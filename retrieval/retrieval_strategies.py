"""
Retrieval strategies module for different document retrieval approaches
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalStrategy:
    """Base class for retrieval strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def retrieve(self, query: str, documents: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query"""
        raise NotImplementedError


class CosineSimilarityRetrieval(RetrievalStrategy):
    """Simple cosine similarity retrieval"""
    
    def __init__(self, embeddings: List[List[float]]):
        super().__init__("Cosine Similarity")
        self.embeddings = np.array(embeddings)
        # Normalize embeddings for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
    
    def retrieve(self, query_embedding: List[float], documents: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve documents using cosine similarity"""
        try:
            # Normalize query embedding
            query_emb = np.array(query_embedding).reshape(1, -1)
            query_emb = query_emb / np.linalg.norm(query_emb)
            
            # Calculate similarities
            similarities = cosine_similarity(query_emb, self.embeddings)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > config.RETRIEVAL_CONFIG["similarity_threshold"]:
                    results.append((documents[idx], float(similarities[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cosine similarity retrieval: {e}")
            return []


class BM25Retrieval(RetrievalStrategy):
    """BM25-based retrieval for keyword matching"""
    
    def __init__(self, documents: List[str]):
        super().__init__("BM25")
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
    
    def retrieve(self, query: str, documents: List[str] = None, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve documents using BM25"""
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # BM25 scores can be 0 or negative
                    results.append((self.documents[idx], float(scores[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return []


class HybridRetrieval(RetrievalStrategy):
    """Hybrid retrieval combining semantic and keyword search"""
    
    def __init__(self, embeddings: List[List[float]], documents: List[str], 
                 semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        super().__init__("Hybrid")
        self.semantic_retrieval = CosineSimilarityRetrieval(embeddings)
        self.keyword_retrieval = BM25Retrieval(documents)
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.documents = documents
    
    def retrieve(self, query: str, query_embedding: List[float], documents: List[str] = None, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve documents using hybrid approach"""
        try:
            # Get semantic results
            semantic_results = self.semantic_retrieval.retrieve(query_embedding, self.documents, k*2)
            
            # Get keyword results
            keyword_results = self.keyword_retrieval.retrieve(query, self.documents, k*2)
            
            # Combine scores
            combined_scores = {}
            
            # Add semantic scores
            for doc, score in semantic_results:
                combined_scores[doc] = self.semantic_weight * score
            
            # Add keyword scores (normalize BM25 scores)
            if keyword_results:
                max_bm25 = max(score for _, score in keyword_results)
                for doc, score in keyword_results:
                    normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                    if doc in combined_scores:
                        combined_scores[doc] += self.keyword_weight * normalized_score
                    else:
                        combined_scores[doc] = self.keyword_weight * normalized_score
            
            # Sort by combined score and return top k
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []


class Reranker:
    """Document reranking using cross-encoder models"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the reranking model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranking model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading reranking model: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Rerank documents based on query-document relevance"""
        if not self.model or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        try:
            # Create query-document pairs
            pairs = [(query, doc) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Combine documents with scores and sort
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]]


class RetrievalPipeline:
    """Complete retrieval pipeline with multiple strategies"""
    
    def __init__(self, embedding_model, documents: List[str], embeddings: List[List[float]]):
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = embeddings
        
        # Initialize retrieval strategies
        self.strategies = {
            "cosine": CosineSimilarityRetrieval(embeddings),
            "bm25": BM25Retrieval(documents),
            "hybrid": HybridRetrieval(embeddings, documents)
        }
        
        # Initialize reranker
        self.reranker = Reranker()
    
    def retrieve(self, query: str, strategy: str = "hybrid", k: int = 5, 
                 use_reranker: bool = True) -> List[Tuple[str, float]]:
        """Retrieve documents using specified strategy"""
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve documents
        if strategy == "hybrid":
            results = self.strategies[strategy].retrieve(query, query_embedding, self.documents, k*2)
        elif strategy == "cosine":
            results = self.strategies[strategy].retrieve(query_embedding, self.documents, k*2)
        else:  # bm25
            results = self.strategies[strategy].retrieve(query, self.documents, k*2)
        
        # Apply reranker if requested
        if use_reranker and results:
            doc_texts = [doc for doc, _ in results]
            reranked_results = self.reranker.rerank(query, doc_texts, k)
            return reranked_results
        
        return results[:k]
    
    def compare_strategies(self, query: str, k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Compare different retrieval strategies"""
        results = {}
        
        for strategy_name in self.strategies.keys():
            try:
                results[strategy_name] = self.retrieve(query, strategy_name, k, use_reranker=False)
            except Exception as e:
                logger.error(f"Error with strategy {strategy_name}: {e}")
                results[strategy_name] = []
        
        return results


if __name__ == "__main__":
    # Test retrieval strategies
    from embeddings.embedding_models import EmbeddingFactory
    
    # Create embedding model
    embedding_model = EmbeddingFactory.create_embedding_model("huggingface")
    
    # Sample documents
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing deals with understanding and generating human language.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Reinforcement learning involves agents learning through interaction with environment."
    ]
    
    # Generate embeddings
    embeddings = embedding_model.embed_documents(sample_docs)
    
    # Create retrieval pipeline
    pipeline = RetrievalPipeline(embedding_model, sample_docs, embeddings)
    
    # Test query
    query = "What is machine learning?"
    
    # Compare strategies
    results = pipeline.compare_strategies(query, k=3)
    
    print(f"Query: {query}")
    print("\nResults by strategy:")
    for strategy, docs in results.items():
        print(f"\n{strategy.upper()}:")
        for doc, score in docs:
            print(f"  Score: {score:.3f} - {doc}")
    
    # Test with reranker
    print(f"\nWith reranker:")
    reranked_results = pipeline.retrieve(query, "hybrid", k=3, use_reranker=True)
    for doc, score in reranked_results:
        print(f"  Score: {score:.3f} - {doc}")

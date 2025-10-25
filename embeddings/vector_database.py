"""
Vector database module for storing and retrieving embeddings
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
import faiss
import pickle
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """Base class for vector databases"""
    
    def __init__(self, name: str):
        self.name = name
        self.db = None
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents with embeddings to the database"""
        raise NotImplementedError
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save the database to disk"""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load the database from disk"""
        raise NotImplementedError


class ChromaVectorDB(VectorDatabase):
    """ChromaDB implementation"""
    
    def __init__(self, collection_name: str = "research_papers"):
        super().__init__("ChromaDB")
        self.collection_name = collection_name
        self.persist_directory = config.VECTOR_DB_CONFIG["chroma"]["persist_directory"]
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents to ChromaDB"""
        try:
            # Prepare data for ChromaDB
            ids = [f"doc_{i}" for i in range(len(documents))]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents in ChromaDB"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            documents_with_scores = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    documents_with_scores.append((doc, similarity_score))
            
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def save(self, path: str = None):
        """ChromaDB auto-persists, so this is a no-op"""
        logger.info("ChromaDB automatically persists data")
    
    def load(self, path: str = None):
        """ChromaDB auto-loads, so this is a no-op"""
        logger.info("ChromaDB automatically loads data")


class FAISSVectorDB(VectorDatabase):
    """FAISS implementation"""
    
    def __init__(self):
        super().__init__("FAISS")
        self.index = None
        self.documents = []
        self.metadatas = []
        self.dimension = None
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Add documents to FAISS index"""
        try:
            if not embeddings:
                logger.error("No embeddings provided")
                return
            
            # Set dimension from first embedding
            if self.dimension is None:
                self.dimension = len(embeddings[0])
                # Create FAISS index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            self.index.add(embeddings_array)
            
            # Store documents and metadata
            self.documents.extend(documents)
            self.metadatas.extend([doc.metadata for doc in documents])
            
            logger.info(f"Added {len(documents)} documents to FAISS index")
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents in FAISS"""
        try:
            if self.index is None:
                logger.error("FAISS index not initialized")
                return []
            
            # Convert query to numpy array and normalize
            query_array = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, k)
            
            documents_with_scores = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    documents_with_scores.append((doc, float(score)))
            
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return []
    
    def save(self, path: str):
        """Save FAISS index and documents to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.faiss")
            
            # Save documents and metadata
            data = {
                "documents": self.documents,
                "metadatas": self.metadatas,
                "dimension": self.dimension
            }
            
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved FAISS index to {path}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def load(self, path: str):
        """Load FAISS index and documents from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.faiss")
            
            # Load documents and metadata
            with open(f"{path}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]
            self.dimension = data["dimension"]
            
            logger.info(f"Loaded FAISS index from {path}")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")


class VectorDBFactory:
    """Factory class for creating vector databases"""
    
    @staticmethod
    def create_vector_db(db_type: str, **kwargs) -> VectorDatabase:
        """Create a vector database based on type"""
        
        if db_type.lower() == "chroma":
            collection_name = kwargs.get("collection_name", "research_papers")
            return ChromaVectorDB(collection_name)
        
        elif db_type.lower() == "faiss":
            return FAISSVectorDB()
        
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
    
    @staticmethod
    def get_available_dbs() -> List[str]:
        """Get list of available vector databases"""
        return ["chroma", "faiss"]


class VectorDBManager:
    """Manager class for vector database operations"""
    
    def __init__(self, db_type: str = "chroma"):
        self.db_type = db_type
        self.vector_db = VectorDBFactory.create_vector_db(db_type)
        self.embedding_model = None
    
    def set_embedding_model(self, embedding_model):
        """Set the embedding model to use"""
        self.embedding_model = embedding_model
    
    def index_documents(self, documents: List[Document]):
        """Index documents using the embedding model"""
        if not self.embedding_model:
            raise ValueError("Embedding model not set")
        
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Add to vector database
        self.vector_db.add_documents(documents, embeddings)
        
        logger.info(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if not self.embedding_model:
            raise ValueError("Embedding model not set")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector database
        return self.vector_db.similarity_search(query_embedding, k)
    
    def save(self, path: str = None):
        """Save the vector database"""
        if path is None:
            if self.db_type == "chroma":
                path = config.VECTOR_DB_CONFIG["chroma"]["persist_directory"]
            else:
                path = config.VECTOR_DB_CONFIG["faiss"]["index_path"]
        
        self.vector_db.save(path)
    
    def load(self, path: str = None):
        """Load the vector database"""
        if path is None:
            if self.db_type == "chroma":
                path = config.VECTOR_DB_CONFIG["chroma"]["persist_directory"]
            else:
                path = config.VECTOR_DB_CONFIG["faiss"]["index_path"]
        
        self.vector_db.load(path)


if __name__ == "__main__":
    # Test vector database
    from embeddings.embedding_models import EmbeddingFactory
    
    # Create embedding model
    embedding_model = EmbeddingFactory.create_embedding_model("huggingface")
    
    # Create vector database manager
    db_manager = VectorDBManager("chroma")
    db_manager.set_embedding_model(embedding_model)
    
    # Test with sample documents
    from langchain.schema import Document
    
    sample_docs = [
        Document(page_content="This is about machine learning algorithms.", metadata={"source": "test1"}),
        Document(page_content="This discusses deep learning and neural networks.", metadata={"source": "test2"}),
        Document(page_content="This covers natural language processing techniques.", metadata={"source": "test3"})
    ]
    
    # Index documents
    db_manager.index_documents(sample_docs)
    
    # Search
    results = db_manager.search("What is machine learning?", k=2)
    
    print(f"Found {len(results)} results:")
    for doc, score in results:
        print(f"Score: {score:.3f}, Content: {doc.page_content[:50]}...")

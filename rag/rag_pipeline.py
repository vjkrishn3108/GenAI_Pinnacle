"""
RAG Pipeline module for Retrieval-Augmented Generation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline class"""
    
    def __init__(self, embedding_model, vector_db_manager, retrieval_pipeline):
        self.embedding_model = embedding_model
        self.vector_db_manager = vector_db_manager
        self.retrieval_pipeline = retrieval_pipeline
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create LLM chain
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found")
            
            return ChatOpenAI(
                model_name=config.LLM_CONFIG["model_name"],
                temperature=config.LLM_CONFIG["temperature"],
                max_tokens=config.LLM_CONFIG["max_tokens"],
                openai_api_key=config.OPENAI_API_KEY
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _create_prompt_template(self):
        """Create the prompt template for RAG"""
        template = """You are an AI assistant that answers questions based on research papers about AI and Data Science. 
Use the following context documents to answer the question. If the answer cannot be found in the context, 
say so clearly.

Context Documents:
{context}

Question: {question}

Answer: Provide a comprehensive answer based on the context documents. Include relevant details and explanations. 
If you reference specific information, mention which document it came from.

Sources: List the source documents used in your answer."""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def retrieve_context(self, query: str, k: int = 5, strategy: str = "hybrid") -> List[Tuple[Document, float]]:
        """Retrieve relevant context documents"""
        try:
            # Use retrieval pipeline to get documents
            results = self.retrieval_pipeline.retrieve(query, strategy, k, use_reranker=True)
            
            # Convert to Document objects if needed
            context_docs = []
            for doc_text, score in results:
                doc = Document(
                    page_content=doc_text,
                    metadata={"similarity_score": score}
                )
                context_docs.append((doc, score))
            
            return context_docs
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """Generate response using retrieved context"""
        try:
            if not context_docs:
                return {
                    "answer": "I couldn't find relevant information in the research papers to answer your question.",
                    "sources": [],
                    "context_used": []
                }
            
            # Prepare context
            context_text = "\n\n".join([
                f"Document {i+1} (Score: {score:.3f}):\n{doc.page_content}"
                for i, (doc, score) in enumerate(context_docs)
            ])
            
            # Generate response
            response = self.llm_chain.run(context=context_text, question=query)
            
            # Extract sources
            sources = []
            context_used = []
            
            for doc, score in context_docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "similarity_score": score,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
                context_used.append(doc.page_content)
            
            return {
                "answer": response,
                "sources": sources,
                "context_used": context_used,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "context_used": []
            }
    
    def ask_question(self, query: str, k: int = 5, strategy: str = "hybrid") -> Dict[str, Any]:
        """Main method to ask a question and get an answer"""
        logger.info(f"Processing question: {query}")
        
        # Retrieve context
        context_docs = self.retrieve_context(query, k, strategy)
        
        # Generate response
        response = self.generate_response(query, context_docs)
        
        # Add query info
        response["query"] = query
        response["strategy_used"] = strategy
        response["k_retrieved"] = k
        
        return response
    
    def compare_strategies(self, query: str, k: int = 5) -> Dict[str, Dict[str, Any]]:
        """Compare different retrieval strategies for the same query"""
        strategies = ["cosine", "bm25", "hybrid"]
        results = {}
        
        for strategy in strategies:
            try:
                results[strategy] = self.ask_question(query, k, strategy)
            except Exception as e:
                logger.error(f"Error with strategy {strategy}: {e}")
                results[strategy] = {
                    "answer": f"Error with strategy {strategy}: {str(e)}",
                    "sources": [],
                    "query": query,
                    "strategy_used": strategy
                }
        
        return results


class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.test_queries = [
            "What is transformer architecture?",
            "How does attention mechanism work in neural networks?",
            "What are the applications of large language models?",
            "Explain the concept of fine-tuning in machine learning",
            "What is the difference between supervised and unsupervised learning?",
            "How do convolutional neural networks work?",
            "What is reinforcement learning?",
            "Explain the concept of transfer learning",
            "What are the challenges in natural language processing?",
            "How does gradient descent optimization work?"
        ]
    
    def evaluate_query(self, query: str, expected_keywords: List[str] = None) -> Dict[str, Any]:
        """Evaluate a single query"""
        try:
            response = self.rag_pipeline.ask_question(query)
            
            # Basic metrics
            evaluation = {
                "query": query,
                "answer_length": len(response["answer"]),
                "num_sources": response["num_sources"],
                "avg_similarity_score": 0,
                "has_answer": len(response["answer"]) > 50,
                "strategy_used": response["strategy_used"]
            }
            
            # Calculate average similarity score
            if response["sources"]:
                scores = [source["similarity_score"] for source in response["sources"]]
                evaluation["avg_similarity_score"] = sum(scores) / len(scores)
            
            # Check for expected keywords
            if expected_keywords:
                answer_lower = response["answer"].lower()
                found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
                evaluation["keyword_coverage"] = len(found_keywords) / len(expected_keywords)
                evaluation["found_keywords"] = found_keywords
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            return {
                "query": query,
                "error": str(e),
                "has_answer": False
            }
    
    def evaluate_system(self) -> Dict[str, Any]:
        """Evaluate the entire RAG system"""
        results = []
        
        for query in self.test_queries:
            evaluation = self.evaluate_query(query)
            results.append(evaluation)
        
        # Calculate overall metrics
        total_queries = len(results)
        successful_queries = sum(1 for r in results if r.get("has_answer", False))
        avg_answer_length = sum(r.get("answer_length", 0) for r in results) / total_queries
        avg_sources = sum(r.get("num_sources", 0) for r in results) / total_queries
        avg_similarity = sum(r.get("avg_similarity_score", 0) for r in results) / total_queries
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries,
            "avg_answer_length": avg_answer_length,
            "avg_sources_per_query": avg_sources,
            "avg_similarity_score": avg_similarity,
            "detailed_results": results
        }
    
    def compare_strategies_evaluation(self) -> Dict[str, Any]:
        """Compare different strategies using evaluation metrics"""
        strategy_results = {}
        
        strategies = ["cosine", "bm25", "hybrid"]
        
        for strategy in strategies:
            results = []
            for query in self.test_queries[:5]:  # Test with first 5 queries
                try:
                    response = self.rag_pipeline.ask_question(query, strategy=strategy)
                    evaluation = self.evaluate_query(query)
                    evaluation["strategy"] = strategy
                    results.append(evaluation)
                except Exception as e:
                    logger.error(f"Error with strategy {strategy} for query {query}: {e}")
            
            # Calculate metrics for this strategy
            if results:
                successful = sum(1 for r in results if r.get("has_answer", False))
                avg_similarity = sum(r.get("avg_similarity_score", 0) for r in results) / len(results)
                
                strategy_results[strategy] = {
                    "success_rate": successful / len(results),
                    "avg_similarity_score": avg_similarity,
                    "num_queries": len(results)
                }
        
        return strategy_results


if __name__ == "__main__":
    # Test RAG pipeline
    from embeddings.embedding_models import EmbeddingFactory
    from embeddings.vector_database import VectorDBManager
    from retrieval.retrieval_strategies import RetrievalPipeline
    
    # Create components
    embedding_model = EmbeddingFactory.create_embedding_model("huggingface")
    
    # Sample documents for testing
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Deep learning uses neural networks with multiple layers to process data and learn complex patterns.",
        "Natural language processing deals with understanding and generating human language using computers.",
        "Computer vision enables machines to interpret and understand visual information from images and videos.",
        "Reinforcement learning involves agents learning through interaction with their environment and receiving rewards."
    ]
    
    # Generate embeddings
    embeddings = embedding_model.embed_documents(sample_docs)
    
    # Create retrieval pipeline
    retrieval_pipeline = RetrievalPipeline(embedding_model, sample_docs, embeddings)
    
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(embedding_model, None, retrieval_pipeline)
    
    # Test query
    query = "What is machine learning?"
    response = rag_pipeline.ask_question(query)
    
    print(f"Query: {query}")
    print(f"Answer: {response['answer']}")
    print(f"Sources: {len(response['sources'])}")
    print(f"Strategy: {response['strategy_used']}")

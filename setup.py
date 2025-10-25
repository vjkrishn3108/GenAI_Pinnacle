"""
Setup script for the Research Paper Answer Bot
This script handles the complete setup and testing of the RAG system
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from utils.document_processor import DocumentProcessor
from embeddings.embedding_models import EmbeddingFactory, EmbeddingEvaluator
from embeddings.vector_database import VectorDBManager
from retrieval.retrieval_strategies import RetrievalPipeline
from rag.rag_pipeline import RAGPipeline, RAGEvaluator
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        config.DATA_PATHS["raw_documents"],
        config.DATA_PATHS["processed_documents"],
        config.DATA_PATHS["embeddings"],
        config.DATA_PATHS["logs"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def process_documents():
    """Process research papers into chunks"""
    logger.info("Processing documents...")
    
    processor = DocumentProcessor()
    
    # Check if documents exist
    raw_dir = config.DATA_PATHS["raw_documents"]
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        logger.error(f"No documents found in {raw_dir}")
        return False
    
    # Process documents
    documents = processor.process_directory(raw_dir)
    
    if not documents:
        logger.error("No documents were processed")
        return False
    
    # Save processed documents
    output_path = os.path.join(config.DATA_PATHS["processed_documents"], "processed_docs.pkl")
    processor.save_processed_documents(documents, output_path)
    
    logger.info(f"Successfully processed {len(documents)} document chunks")
    return True

def test_embedding_models():
    """Test different embedding models"""
    logger.info("Testing embedding models...")
    
    # Test HuggingFace model (no API key required)
    try:
        hf_model = EmbeddingFactory.create_embedding_model("huggingface")
        logger.info(f"HuggingFace model loaded: {hf_model.model_name}")
        
        # Test embedding
        test_texts = [
            "This is a test document about machine learning.",
            "Another test document about deep learning."
        ]
        embeddings = hf_model.embed_documents(test_texts)
        logger.info(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
        
        return hf_model
        
    except Exception as e:
        logger.error(f"Error testing HuggingFace model: {e}")
        return None

def setup_vector_database(embedding_model):
    """Set up vector database"""
    logger.info("Setting up vector database...")
    
    try:
        # Create vector database manager
        db_manager = VectorDBManager("chroma")
        db_manager.set_embedding_model(embedding_model)
        
        # Load processed documents
        processor = DocumentProcessor()
        processed_path = os.path.join(config.DATA_PATHS["processed_documents"], "processed_docs.pkl")
        
        if os.path.exists(processed_path):
            documents = processor.load_processed_documents(processed_path)
            
            if documents:
                # Index documents
                db_manager.index_documents(documents)
                logger.info(f"Indexed {len(documents)} documents in vector database")
                return db_manager, documents
            else:
                logger.error("No documents loaded from processed file")
                return None, None
        else:
            logger.error("Processed documents file not found")
            return None, None
            
    except Exception as e:
        logger.error(f"Error setting up vector database: {e}")
        return None, None

def setup_retrieval_pipeline(embedding_model, documents):
    """Set up retrieval pipeline"""
    logger.info("Setting up retrieval pipeline...")
    
    try:
        # Prepare texts and embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = embedding_model.embed_documents(texts)
        
        # Create retrieval pipeline
        retrieval_pipeline = RetrievalPipeline(embedding_model, texts, embeddings)
        
        logger.info("Retrieval pipeline created successfully")
        return retrieval_pipeline
        
    except Exception as e:
        logger.error(f"Error setting up retrieval pipeline: {e}")
        return None

def test_rag_system(embedding_model, retrieval_pipeline):
    """Test the complete RAG system"""
    logger.info("Testing RAG system...")
    
    try:
        # Create RAG pipeline
        rag_pipeline = RAGPipeline(embedding_model, None, retrieval_pipeline)
        
        # Test queries
        test_queries = [
            "What is transformer architecture?",
            "How does attention mechanism work?",
            "What are the applications of large language models?"
        ]
        
        logger.info("Running test queries...")
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            
            try:
                response = rag_pipeline.ask_question(query, k=3)
                
                logger.info(f"Answer length: {len(response['answer'])}")
                logger.info(f"Sources used: {response['num_sources']}")
                logger.info(f"Strategy: {response['strategy_used']}")
                
                # Show first part of answer
                answer_preview = response['answer'][:200] + "..." if len(response['answer']) > 200 else response['answer']
                logger.info(f"Answer preview: {answer_preview}")
                
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
        
        return rag_pipeline
        
    except Exception as e:
        logger.error(f"Error testing RAG system: {e}")
        return None

def run_evaluation(rag_pipeline):
    """Run system evaluation"""
    logger.info("Running system evaluation...")
    
    try:
        evaluator = RAGEvaluator(rag_pipeline)
        evaluation_results = evaluator.evaluate_system()
        
        logger.info("=== EVALUATION RESULTS ===")
        logger.info(f"Total queries: {evaluation_results['total_queries']}")
        logger.info(f"Successful queries: {evaluation_results['successful_queries']}")
        logger.info(f"Success rate: {evaluation_results['success_rate']:.1%}")
        logger.info(f"Average answer length: {evaluation_results['avg_answer_length']:.0f}")
        logger.info(f"Average sources per query: {evaluation_results['avg_sources_per_query']:.1f}")
        logger.info(f"Average similarity score: {evaluation_results['avg_similarity_score']:.3f}")
        
        # Strategy comparison
        strategy_comparison = evaluator.compare_strategies_evaluation()
        if strategy_comparison:
            logger.info("\n=== STRATEGY COMPARISON ===")
            for strategy, metrics in strategy_comparison.items():
                logger.info(f"{strategy.upper()}:")
                logger.info(f"  Success rate: {metrics['success_rate']:.1%}")
                logger.info(f"  Avg similarity: {metrics['avg_similarity_score']:.3f}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return None

def main():
    """Main setup function"""
    logger.info("Starting Research Paper Answer Bot setup...")
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Process documents
    if not process_documents():
        logger.error("Document processing failed. Exiting.")
        return False
    
    # Step 3: Test embedding models
    embedding_model = test_embedding_models()
    if not embedding_model:
        logger.error("Embedding model setup failed. Exiting.")
        return False
    
    # Step 4: Setup vector database
    db_manager, documents = setup_vector_database(embedding_model)
    if not db_manager or not documents:
        logger.error("Vector database setup failed. Exiting.")
        return False
    
    # Step 5: Setup retrieval pipeline
    retrieval_pipeline = setup_retrieval_pipeline(embedding_model, documents)
    if not retrieval_pipeline:
        logger.error("Retrieval pipeline setup failed. Exiting.")
        return False
    
    # Step 6: Test RAG system
    rag_pipeline = test_rag_system(embedding_model, retrieval_pipeline)
    if not rag_pipeline:
        logger.error("RAG system test failed. Exiting.")
        return False
    
    # Step 7: Run evaluation
    evaluation_results = run_evaluation(rag_pipeline)
    if evaluation_results:
        logger.info("Setup completed successfully!")
        logger.info("You can now run the Streamlit app with: streamlit run app.py")
        return True
    else:
        logger.error("Evaluation failed.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Setup completed successfully!")
        print("📚 Research Paper Answer Bot is ready!")
        print("🚀 Run 'streamlit run app.py' to start the web interface")
    else:
        print("\n❌ Setup failed. Please check the logs for errors.")

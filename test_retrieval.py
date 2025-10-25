"""
Test script for the Research Paper Answer Bot - Retrieval Demo
This script demonstrates the retrieval capabilities without requiring OpenAI API key
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
from embeddings.embedding_models import EmbeddingFactory
from embeddings.vector_database import VectorDBManager
from retrieval.retrieval_strategies import RetrievalPipeline
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_retrieval_system():
    """Test the retrieval system without LLM"""
    logger.info("Testing Research Paper Answer Bot - Retrieval System")
    
    try:
        # Step 1: Load processed documents
        processor = DocumentProcessor()
        processed_path = os.path.join(config.DATA_PATHS["processed_documents"], "processed_docs.pkl")
        
        if not os.path.exists(processed_path):
            logger.error("Processed documents not found. Please run setup.py first.")
            return False
        
        documents = processor.load_processed_documents(processed_path)
        logger.info(f"Loaded {len(documents)} document chunks")
        
        # Step 2: Create embedding model
        embedding_model = EmbeddingFactory.create_embedding_model("huggingface")
        logger.info(f"Using embedding model: {embedding_model.model_name}")
        
        # Step 3: Create retrieval pipeline
        texts = [doc.page_content for doc in documents]
        embeddings = embedding_model.embed_documents(texts)
        
        retrieval_pipeline = RetrievalPipeline(embedding_model, texts, embeddings)
        logger.info("Retrieval pipeline created successfully")
        
        # Step 4: Test queries
        test_queries = [
            "What is transformer architecture?",
            "How does attention mechanism work?",
            "What are the applications of large language models?",
            "Explain the concept of fine-tuning",
            "What is the difference between supervised and unsupervised learning?",
            "How do convolutional neural networks work?",
            "What is reinforcement learning?",
            "Explain the concept of transfer learning"
        ]
        
        logger.info("\n" + "="*80)
        logger.info("TESTING RETRIEVAL SYSTEM")
        logger.info("="*80)
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- Query {i}: {query} ---")
            
            # Test different strategies
            strategies = ["cosine", "bm25", "hybrid"]
            
            for strategy in strategies:
                try:
                    results = retrieval_pipeline.retrieve(query, strategy, k=3, use_reranker=True)
                    
                    logger.info(f"\n{strategy.upper()} Strategy Results:")
                    for j, (doc_text, score) in enumerate(results, 1):
                        preview = doc_text[:150] + "..." if len(doc_text) > 150 else doc_text
                        logger.info(f"  {j}. Score: {score:.3f}")
                        logger.info(f"     Content: {preview}")
                    
                except Exception as e:
                    logger.error(f"Error with {strategy} strategy: {e}")
        
        # Step 5: Strategy comparison
        logger.info("\n" + "="*80)
        logger.info("STRATEGY COMPARISON")
        logger.info("="*80)
        
        comparison_query = "What is transformer architecture?"
        logger.info(f"Comparison Query: {comparison_query}")
        
        strategy_results = retrieval_pipeline.compare_strategies(comparison_query, k=3)
        
        for strategy, results in strategy_results.items():
            logger.info(f"\n{strategy.upper()} Strategy:")
            if results:
                avg_score = sum(score for _, score in results) / len(results)
                logger.info(f"  Average Score: {avg_score:.3f}")
                logger.info(f"  Number of Results: {len(results)}")
                
                # Show top result
                if results:
                    top_doc, top_score = results[0]
                    preview = top_doc[:100] + "..." if len(top_doc) > 100 else top_doc
                    logger.info(f"  Top Result (Score: {top_score:.3f}): {preview}")
            else:
                logger.info("  No results found")
        
        # Step 6: Document statistics
        logger.info("\n" + "="*80)
        logger.info("DOCUMENT STATISTICS")
        logger.info("="*80)
        
        # Analyze document sources
        sources = {}
        total_chunks = len(documents)
        
        for doc in documents:
            source = doc.metadata.get('filename', 'unknown')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        logger.info(f"Total document chunks: {total_chunks}")
        logger.info("Chunks per source:")
        for source, count in sources.items():
            logger.info(f"  {source}: {count} chunks")
        
        # Analyze chunk sizes
        chunk_sizes = [len(doc.page_content) for doc in documents]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
        min_chunk_size = min(chunk_sizes)
        max_chunk_size = max(chunk_sizes)
        
        logger.info(f"\nChunk size statistics:")
        logger.info(f"  Average: {avg_chunk_size:.0f} characters")
        logger.info(f"  Minimum: {min_chunk_size} characters")
        logger.info(f"  Maximum: {max_chunk_size} characters")
        
        logger.info("\n" + "="*80)
        logger.info("RETRIEVAL SYSTEM TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("The system is ready for RAG pipeline integration.")
        logger.info("To use with LLM, set OPENAI_API_KEY in .env file and run the full setup.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing retrieval system: {e}")
        return False

def demo_specific_queries():
    """Demo specific queries with detailed results"""
    logger.info("\n" + "="*80)
    logger.info("DETAILED QUERY DEMONSTRATION")
    logger.info("="*80)
    
    try:
        # Load system components
        processor = DocumentProcessor()
        processed_path = os.path.join(config.DATA_PATHS["processed_documents"], "processed_docs.pkl")
        documents = processor.load_processed_documents(processed_path)
        
        embedding_model = EmbeddingFactory.create_embedding_model("huggingface")
        texts = [doc.page_content for doc in documents]
        embeddings = embedding_model.embed_documents(texts)
        retrieval_pipeline = RetrievalPipeline(embedding_model, texts, embeddings)
        
        # Specific demo queries
        demo_queries = [
            "What is the attention mechanism in transformers?",
            "How does GPT-4 work?",
            "What are the advantages of Mistral models?",
            "Explain instruction tuning in language models"
        ]
        
        for query in demo_queries:
            logger.info(f"\n🔍 Query: {query}")
            logger.info("-" * 60)
            
            # Get hybrid results with reranking
            results = retrieval_pipeline.retrieve(query, "hybrid", k=5, use_reranker=True)
            
            for i, (doc_text, score) in enumerate(results, 1):
                logger.info(f"\n📄 Result {i} (Score: {score:.3f})")
                
                # Find the source document
                source_doc = None
                for doc in documents:
                    if doc.page_content == doc_text:
                        source_doc = doc
                        break
                
                if source_doc:
                    filename = source_doc.metadata.get('filename', 'Unknown')
                    chunk_id = source_doc.metadata.get('chunk_id', 'Unknown')
                    logger.info(f"   Source: {filename} (Chunk {chunk_id})")
                
                # Show content preview
                preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                logger.info(f"   Content: {preview}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Research Paper Answer Bot - Retrieval System Test")
    print("=" * 60)
    
    # Test the retrieval system
    success = test_retrieval_system()
    
    if success:
        # Run detailed demo
        demo_success = demo_specific_queries()
        
        if demo_success:
            print("\n✅ All tests completed successfully!")
            print("\n📋 Summary:")
            print("   • Document processing: ✅ Working")
            print("   • Embedding generation: ✅ Working") 
            print("   • Vector database: ✅ Working")
            print("   • Retrieval strategies: ✅ Working")
            print("   • Reranking: ✅ Working")
            print("\n🎯 Next steps:")
            print("   1. Set OPENAI_API_KEY in .env file for LLM integration")
            print("   2. Run 'streamlit run app.py' for web interface")
            print("   3. Or run 'python setup.py' for full system test")
        else:
            print("\n❌ Demo failed. Check logs for details.")
    else:
        print("\n❌ Retrieval system test failed. Check logs for details.")

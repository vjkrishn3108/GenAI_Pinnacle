"""
Streamlit application for the Research Paper Answer Bot
"""

import streamlit as st
import logging
import os
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our modules
from utils.document_processor import DocumentProcessor
from embeddings.embedding_models import EmbeddingFactory, EmbeddingEvaluator
from embeddings.vector_database import VectorDBManager
from retrieval.retrieval_strategies import RetrievalPipeline
from rag.rag_pipeline import RAGPipeline, RAGEvaluator
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'vector_db_manager' not in st.session_state:
    st.session_state.vector_db_manager = None
if 'retrieval_pipeline' not in st.session_state:
    st.session_state.retrieval_pipeline = None

def initialize_system():
    """Initialize the RAG system components"""
    try:
        # Create embedding model
        embedding_model = EmbeddingFactory.create_embedding_model("huggingface")
        st.session_state.embedding_model = embedding_model
        
        # Create vector database manager
        vector_db_manager = VectorDBManager("chroma")
        vector_db_manager.set_embedding_model(embedding_model)
        st.session_state.vector_db_manager = vector_db_manager
        
        st.success("System initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return False

def load_documents():
    """Load and process documents"""
    try:
        processor = DocumentProcessor()
        
        # Check if processed documents exist
        processed_path = os.path.join(config.DATA_PATHS["processed_documents"], "processed_docs.pkl")
        
        if os.path.exists(processed_path):
            documents = processor.load_processed_documents(processed_path)
        else:
            # Process raw documents
            documents = processor.process_directory(config.DATA_PATHS["raw_documents"])
            if documents:
                processor.save_processed_documents(documents, processed_path)
        
        if documents:
            # Index documents
            st.session_state.vector_db_manager.index_documents(documents)
            
            # Create retrieval pipeline
            texts = [doc.page_content for doc in documents]
            embeddings = st.session_state.embedding_model.embed_documents(texts)
            retrieval_pipeline = RetrievalPipeline(
                st.session_state.embedding_model, 
                texts, 
                embeddings
            )
            st.session_state.retrieval_pipeline = retrieval_pipeline
            
            # Create RAG pipeline
            rag_pipeline = RAGPipeline(
                st.session_state.embedding_model,
                st.session_state.vector_db_manager,
                retrieval_pipeline
            )
            st.session_state.rag_pipeline = rag_pipeline
            st.session_state.documents_loaded = True
            
            st.success(f"Successfully loaded {len(documents)} document chunks!")
            return len(documents)
        else:
            st.warning("No documents found. Please add PDF files to the data/raw directory.")
            return 0
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return 0

def display_source_documents(sources: List[Dict[str, Any]]):
    """Display source documents in an attractive format"""
    if not sources:
        st.info("No source documents found.")
        return
    
    st.subheader("📄 Source Documents")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i} (Similarity: {source['similarity_score']:.3f})"):
            st.markdown(f"**Content:**")
            st.write(source['content'])
            
            if 'metadata' in source and source['metadata']:
                st.markdown("**Metadata:**")
                for key, value in source['metadata'].items():
                    st.write(f"- {key}: {value}")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Research Paper Answer Bot</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered Q&A system for research papers on AI and Data Science")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Initialize system button
        if st.button("🚀 Initialize System"):
            with st.spinner("Initializing system..."):
                initialize_system()
        
        # Load documents button
        if st.button("📁 Load Documents"):
            with st.spinner("Loading and processing documents..."):
                num_docs = load_documents()
                if num_docs > 0:
                    st.session_state.documents_loaded = True
        
        # System status
        st.subheader("📊 System Status")
        st.write(f"**Documents Loaded:** {'✅ Yes' if st.session_state.documents_loaded else '❌ No'}")
        st.write(f"**RAG Pipeline:** {'✅ Ready' if st.session_state.rag_pipeline else '❌ Not Ready'}")
        
        # Configuration options
        if st.session_state.documents_loaded:
            st.subheader("🔧 Query Settings")
            
            retrieval_strategy = st.selectbox(
                "Retrieval Strategy",
                ["hybrid", "cosine", "bm25"],
                help="Choose the retrieval strategy for finding relevant documents"
            )
            
            num_sources = st.slider(
                "Number of Sources",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of source documents to retrieve"
            )
            
            use_reranker = st.checkbox(
                "Use Reranker",
                value=True,
                help="Apply reranking to improve document relevance"
            )
    
    # Main content area
    if not st.session_state.documents_loaded:
        st.info("👈 Please initialize the system and load documents from the sidebar to get started.")
        
        # Show sample questions
        st.subheader("💡 Sample Questions You Can Ask")
        sample_questions = [
            "What is transformer architecture?",
            "How does attention mechanism work?",
            "What are the applications of large language models?",
            "Explain the concept of fine-tuning",
            "What is the difference between supervised and unsupervised learning?"
        ]
        
        for question in sample_questions:
            st.write(f"• {question}")
        
        return
    
    # Query interface
    st.subheader("❓ Ask a Question")
    
    # Query input
    query = st.text_input(
        "Enter your question about AI/ML research papers:",
        placeholder="e.g., What is transformer architecture?"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        ask_button = st.button("🔍 Ask Question", type="primary")
    
    # Process query
    if ask_button and query:
        with st.spinner("Processing your question..."):
            try:
                # Get response from RAG pipeline
                response = st.session_state.rag_pipeline.ask_question(
                    query, 
                    k=num_sources, 
                    strategy=retrieval_strategy
                )
                
                # Display answer
                st.subheader("💬 Answer")
                st.markdown(response['answer'])
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Used", response['num_sources'])
                with col2:
                    st.metric("Strategy", response['strategy_used'].title())
                with col3:
                    avg_score = sum(s['similarity_score'] for s in response['sources']) / len(response['sources']) if response['sources'] else 0
                    st.metric("Avg Similarity", f"{avg_score:.3f}")
                
                # Display source documents
                display_source_documents(response['sources'])
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    # Evaluation section
    if st.session_state.rag_pipeline:
        st.subheader("📈 System Evaluation")
        
        if st.button("🧪 Run Evaluation"):
            with st.spinner("Running system evaluation..."):
                try:
                    evaluator = RAGEvaluator(st.session_state.rag_pipeline)
                    evaluation_results = evaluator.evaluate_system()
                    
                    # Display evaluation metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Success Rate", 
                            f"{evaluation_results['success_rate']:.1%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Avg Answer Length", 
                            f"{evaluation_results['avg_answer_length']:.0f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Avg Sources", 
                            f"{evaluation_results['avg_sources_per_query']:.1f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Avg Similarity", 
                            f"{evaluation_results['avg_similarity_score']:.3f}"
                        )
                    
                    # Strategy comparison
                    st.subheader("🔄 Strategy Comparison")
                    strategy_comparison = evaluator.compare_strategies_evaluation()
                    
                    if strategy_comparison:
                        # Create comparison chart
                        strategies = list(strategy_comparison.keys())
                        success_rates = [strategy_comparison[s]['success_rate'] for s in strategies]
                        avg_similarities = [strategy_comparison[s]['avg_similarity_score'] for s in strategies]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            name='Success Rate',
                            x=strategies,
                            y=success_rates,
                            yaxis='y',
                            offsetgroup=1
                        ))
                        fig.add_trace(go.Bar(
                            name='Avg Similarity',
                            x=strategies,
                            y=avg_similarities,
                            yaxis='y2',
                            offsetgroup=2
                        ))
                        
                        fig.update_layout(
                            title='Strategy Performance Comparison',
                            xaxis_title='Strategy',
                            yaxis=dict(title='Success Rate', side='left'),
                            yaxis2=dict(title='Average Similarity', side='right', overlaying='y'),
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running evaluation: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Research Paper Answer Bot** | Built with LangChain, Streamlit, and advanced RAG techniques"
    )

if __name__ == "__main__":
    main()

"""
Document processing module for loading and chunking research papers
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles loading and processing of research papers"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.DOCUMENT_CONFIG["chunk_size"],
            chunk_overlap=config.DOCUMENT_CONFIG["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf_pymupdf(self, file_path: str) -> str:
        """Load PDF using PyMuPDF (faster and more accurate)"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""
    
    def load_pdf_pypdf2(self, file_path: str) -> str:
        """Load PDF using PyPDF2 (fallback method)"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""
    
    def load_document(self, file_path: str) -> str:
        """Load document based on file extension"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            # Try PyMuPDF first, fallback to PyPDF2
            text = self.load_pdf_pymupdf(str(file_path))
            if not text.strip():
                text = self.load_pdf_pypdf2(str(file_path))
            return text
        
        elif file_path.suffix.lower() == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error loading text file {file_path}: {e}")
                return ""
        
        else:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            return ""
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document into chunks"""
        text = self.load_document(file_path)
        if not text.strip():
            return []
        
        # Create metadata
        file_name = Path(file_path).name
        metadata = {
            "source": file_path,
            "filename": file_name,
            "file_type": Path(file_path).suffix.lower()
        }
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        logger.info(f"Processed {file_name}: {len(documents)} chunks")
        return documents
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all supported documents in a directory"""
        all_documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return all_documents
        
        # Find all supported files
        supported_files = []
        for ext in config.DOCUMENT_CONFIG["supported_formats"]:
            supported_files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(supported_files)} supported files")
        
        # Process each file
        for file_path in supported_files:
            documents = self.process_document(str(file_path))
            all_documents.extend(documents)
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents
    
    def save_processed_documents(self, documents: List[Document], output_path: str):
        """Save processed documents to disk"""
        import pickle
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(documents, f)
        
        logger.info(f"Saved {len(documents)} documents to {output_path}")
    
    def load_processed_documents(self, input_path: str) -> List[Document]:
        """Load processed documents from disk"""
        import pickle
        
        try:
            with open(input_path, 'rb') as f:
                documents = pickle.load(f)
            logger.info(f"Loaded {len(documents)} documents from {input_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents from {input_path}: {e}")
            return []


if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    
    # Create sample data directory if it doesn't exist
    os.makedirs(config.DATA_PATHS["raw_documents"], exist_ok=True)
    
    # Process documents
    documents = processor.process_directory(config.DATA_PATHS["raw_documents"])
    
    if documents:
        # Save processed documents
        output_path = os.path.join(config.DATA_PATHS["processed_documents"], "processed_docs.pkl")
        processor.save_processed_documents(documents, output_path)
        print(f"Successfully processed {len(documents)} document chunks")
    else:
        print("No documents found to process. Please add PDF files to the data/raw directory.")

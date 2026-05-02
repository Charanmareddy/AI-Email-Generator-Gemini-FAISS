import os
import tempfile
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Configure logging for the document_loader module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents(uploaded_files: list) -> List[Document]:
    """
    Parses uploaded files (PDF/TXT) into LangChain Document objects.
    
    Args:
        uploaded_files (list): A list of uploaded Streamlit file objects.
        
    Returns:
        List[Document]: A list of LangChain Document objects containing text and metadata.
        
    Example:
        >>> docs = load_documents([pdf_file_obj])
        >>> print(len(docs))
    """
    all_documents = []
    
    for file in uploaded_files:
        # Create a temporary file to save the uploaded content
        # This is required because LangChain loaders need a file path
        try:
            # Use delete=False so we can close the file and pass its path to the loader
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
                temp_file.write(file.getbuffer())
                temp_path = temp_file.name
                
            logger.info(f"Created temporary file for {file.name} at {temp_path}")
            
            # Choose the appropriate loader based on the file extension
            if file.name.endswith('.pdf'):
                # Load PDF file
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
            elif file.name.endswith('.txt'):
                # Load text file
                loader = TextLoader(temp_path, encoding='utf-8')
                docs = loader.load()
            else:
                logger.warning(f"Unsupported file type: {file.name}")
                continue
                
            # Add the source filename to the metadata for citations
            for doc in docs:
                doc.metadata['source'] = file.name
                
            all_documents.extend(docs)
            logger.info(f"Successfully loaded {len(docs)} pages/parts from {file.name}")
            
        except Exception as e:
            logger.error(f"Error loading document {file.name}: {e}")
            raise RuntimeError(f"Failed to load document {file.name}: {str(e)}")
            
        finally:
            # Ensure the temporary file is deleted to free up disk space
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temporary file {temp_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up temporary file {temp_path}: {cleanup_error}")
                    
    return all_documents

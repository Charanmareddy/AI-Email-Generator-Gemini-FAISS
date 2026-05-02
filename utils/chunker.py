import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Configure logging for the chunker module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of LangChain Documents into smaller, manageable chunks.
    
    Args:
        documents (List[Document]): The list of raw Document objects.
        
    Returns:
        List[Document]: A list of chunked Document objects.
        
    Example:
        >>> chunks = chunk_documents(docs)
        >>> print(len(chunks))
    """
    try:
        # We use RecursiveCharacterTextSplitter as it tries to keep paragraphs and sentences together
        # This is critical for maintaining the semantic meaning of the context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Split the documents
        chunked_docs = text_splitter.split_documents(documents)
        logger.info(f"Successfully chunked {len(documents)} documents into {len(chunked_docs)} chunks.")
        
        return chunked_docs
    except Exception as e:
        logger.error(f"Error during document chunking: {e}")
        raise RuntimeError(f"Failed to chunk documents: {str(e)}")

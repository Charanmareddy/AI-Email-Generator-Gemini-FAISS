import logging
from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings

import config

# Configure logging for the embedder module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding_model() -> Any:
    """
    Initializes and returns the local HuggingFace embedding model.
    
    Returns:
        HuggingFaceEmbeddings: The instantiated embedding model object.
        
    Example:
        >>> embedder = get_embedding_model()
        >>> vector = embedder.embed_query("Hello")
    """
    try:
        # We use a lightweight local model to avoid API costs and latency for embeddings
        # model_kwargs={'device': 'cpu'} ensures it works on machines without a GPU
        embedder = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Successfully initialized embedding model: {config.EMBEDDING_MODEL_NAME}")
        return embedder
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")

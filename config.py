import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Configure basic logging for the config module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to load environment variables from the .env file
try:
    # We load variables to ensure API keys are securely accessed
    load_dotenv()
    logger.info("Successfully loaded .env file.")
except Exception as e:
    logger.error(f"Failed to load .env file. Error: {e}")

# Application Constants
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 100
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME: str = "gemini-flash-latest"
TEMPERATURE: float = 0.7

def check_keys() -> bool:
    """
    Validates the presence of required environment variables.
    Supports both local .env files and Streamlit Cloud Secrets.
    
    Returns:
        bool: True if all required keys are present, False otherwise.
        
    Example:
        >>> check_keys()
        True
    """
    try:
        # Try to pull from Streamlit secrets if running in Streamlit Cloud
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                # Langchain expects it in os.environ, so we inject it
                os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
                logger.info("Loaded GEMINI_API_KEY from Streamlit secrets.")
        except Exception:
            # Not running in Streamlit or secrets not configured
            pass

        # Check if the Gemini API key exists and is not empty
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.error("GEMINI_API_KEY is missing or invalid in environment variables or secrets.")
            return False
            
        logger.info("All required environment keys are present.")
        return True
    except Exception as e:
        logger.error(f"Error during key validation: {e}")
        return False

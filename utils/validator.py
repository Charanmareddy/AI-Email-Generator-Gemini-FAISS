import logging
import spacy
from typing import Dict, Any, Optional

# Configure logging for the validator module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to load the spaCy model globally to avoid loading it on every call
try:
    # Load the small English model for basic NLP tasks
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model 'en_core_web_sm'.")
except Exception as e:
    logger.warning(f"Failed to load spaCy model. Will use fallback validation. Error: {e}")
    nlp = None

def validate_input(query: str, uploaded_files: list) -> Dict[str, Any]:
    """
    Validates user query and uploaded files. Enhances the query using NLP (spaCy).
    
    Args:
        query (str): The raw text query from the user.
        uploaded_files (list): A list of uploaded Streamlit file objects.
        
    Returns:
        Dict[str, Any]: A dictionary containing validation status, error messages, 
                        and the processed NLP-enhanced query.
                        
    Example:
        >>> validate_input("Write an email to John.", [file_obj])
        {'is_valid': True, 'error': None, 'enhanced_query': 'Write email John', 'entities': ['John']}
    """
    result = {
        "is_valid": False,
        "error": None,
        "enhanced_query": query,
        "entities": []
    }
    
    try:
        # File upload is now optional, so we skip the empty check.
        # Check if query is empty or too short
        # Check if query is empty or too short
        if not query or len(query.strip()) < 5:
            result["error"] = "Please provide a valid query (at least 5 characters)."
            return result
            
        # NLP Processing: Enhance query by extracting key terms and entities
        if nlp:
            # Process the raw query through the NLP pipeline
            doc = nlp(query)
            
            # Extract named entities (e.g., Person names, Organizations)
            entities = [ent.text for ent in doc.ents]
            result["entities"] = entities
            
            # Extract meaningful words (lemmas of non-stop, non-punct tokens) to form a dense search query
            keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            
            # Combine the original query with the extracted keywords to boost search relevance
            enhanced_query = f"{query} {' '.join(keywords)}"
            result["enhanced_query"] = enhanced_query
            logger.info(f"NLP processing complete. Extracted entities: {entities}")
            
        # If all checks pass, mark as valid
        result["is_valid"] = True
        return result
        
    except Exception as e:
        logger.error(f"Error during input validation: {e}")
        result["error"] = f"An unexpected validation error occurred: {str(e)}"
        return result

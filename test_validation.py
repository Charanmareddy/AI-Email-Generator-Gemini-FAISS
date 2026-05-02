import os
import pytest
from langchain_core.documents import Document

# 1. Test Imports
def test_imports():
    """
    Test 1: Verify all required modules can be imported without errors.
    This ensures the environment is set up correctly according to requirements.txt.
    """
    try:
        import streamlit
        import langchain
        import langchain_google_genai
        import langchain_huggingface
        import faiss
        import spacy
        import dotenv
        # If no ImportError is raised, the test passes
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

# 2. Test Configuration Loading
def test_config():
    """
    Test 2: Verify that configuration constants and the environment key checker work.
    """
    import config
    try:
        # Check if basic constants are defined
        assert hasattr(config, 'CHUNK_SIZE')
        assert hasattr(config, 'LLM_MODEL_NAME')
        
        # We don't fail if the key is missing in CI, we just test the function executes
        # It should return a boolean
        result = config.check_keys()
        assert isinstance(result, bool)
    except Exception as e:
        pytest.fail(f"Config test failed: {e}")

# 3. Test Chunking Logic
def test_chunking():
    """
    Test 3: Verify that the document chunker correctly splits large text.
    """
    from utils.chunker import chunk_documents
    import config
    
    # Temporarily force a small chunk size for testing
    original_size = config.CHUNK_SIZE
    config.CHUNK_SIZE = 50
    config.CHUNK_OVERLAP = 10
    
    try:
        # Create a mock document with 100 characters
        long_text = "A" * 100
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
        
        chunks = chunk_documents(docs)
        
        # Since text is 100 chars and chunk size is 50 with 10 overlap, 
        # we expect more than 1 chunk (roughly 3 chunks)
        assert len(chunks) > 1
        assert chunks[0].page_content == "A" * 50
    except Exception as e:
        pytest.fail(f"Chunking test failed: {e}")
    finally:
        # Restore original config
        config.CHUNK_SIZE = original_size

# 4. Test Embedder Initialization
def test_embedding_model():
    """
    Test 4: Verify that the HuggingFace embedding model initializes without error.
    """
    from utils.embedder import get_embedding_model
    try:
        embedder = get_embedding_model()
        assert embedder is not None
        # Test a simple embedding
        vector = embedder.embed_query("Test query")
        assert len(vector) > 0
    except Exception as e:
        pytest.fail(f"Embedder test failed: {e}")

# 5. Test Validator Logic
def test_validation_logic():
    """
    Test 5: Verify input validation handles edge cases (empty input, missing files).
    """
    from utils.validator import validate_input
    
    try:
        class MockUploadedFile:
            def __init__(self, name):
                self.name = name
                
        # Scenario A: Missing files
        res_no_files = validate_input("Valid query", [])
        assert res_no_files["is_valid"] is False
        assert "upload at least one" in res_no_files["error"].lower()
        
        # Scenario B: Empty query
        res_empty_query = validate_input("", [MockUploadedFile("test.txt")])
        assert res_empty_query["is_valid"] is False
        assert "valid query" in res_empty_query["error"].lower()
        
        # Scenario C: Valid input (NLP processing should run if spacy is loaded)
        res_valid = validate_input("Write an email to John about the project.", [MockUploadedFile("test.txt")])
        assert res_valid["is_valid"] is True
        assert res_valid["error"] is None
        assert "enhanced_query" in res_valid
    except Exception as e:
        pytest.fail(f"Validation test failed: {e}")

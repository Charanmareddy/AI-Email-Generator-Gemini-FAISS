import streamlit as st
import logging

# Import the configuration and validation functions
import config
from utils.validator import validate_input
from utils.document_loader import load_documents
from utils.chunker import chunk_documents
from utils.embedder import get_embedding_model
from utils.retriever import generate_email, generate_email_direct

# Configure logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    """
    The main execution function for the Streamlit application.
    Sets up the UI, handles user input, processes the RAG pipeline, and displays the output.
    """
    # Configure the Streamlit page layout and title
    st.set_page_config(page_title="AI Email Writer", page_icon="📧", layout="wide")
    
    st.title("📧 AI Email Writer (RAG-powered)")
    st.markdown("Upload context documents (PDF/TXT) and let the AI draft a professional email for you.")
    
    # Check for required API keys before proceeding
    if not config.check_keys():
        st.error("Missing or invalid GEMINI_API_KEY. Please check your .env file.")
        st.stop()
        
    # Sidebar for file uploads
    with st.sidebar:
        st.header("1. Upload Context")
        uploaded_files = st.file_uploader(
            "Upload reference documents (PDF or TXT)", 
            type=["pdf", "txt"], 
            accept_multiple_files=True
        )
        st.info("The AI will use these documents to draft your email.")
        
    # Main area for query input
    st.header("2. Email Requirements")
    user_query = st.text_area(
        "What should the email be about?", 
        placeholder="e.g., Write a follow-up email to the client summarizing the project delays mentioned in the report."
    )
    
    # Generate Button
    if st.button("Generate Email", type="primary"):
        # Step 1: Validate Inputs
        validation_result = validate_input(user_query, uploaded_files)
        
        if not validation_result["is_valid"]:
            st.error(validation_result["error"])
            logger.warning(f"Validation failed: {validation_result['error']}")
            return
            
        enhanced_query = validation_result["enhanced_query"]
        st.success(f"Processing query. Extracted intent/entities: {validation_result.get('entities', [])}")
        
        try:
            if uploaded_files:
                # We use a spinner to provide visual feedback while long processes run
                with st.spinner("Loading and processing documents..."):
                    # Step 2: Load Documents
                    raw_docs = load_documents(uploaded_files)
                    
                    # Step 3: Chunk Documents
                    chunked_docs = chunk_documents(raw_docs)
                    
                with st.spinner("Initializing embedding model..."):
                    # Step 4: Initialize Embedder
                    embedder = get_embedding_model()
                    
                with st.spinner("Retrieving context and drafting email..."):
                    # Step 5: Generate Email
                    result = generate_email(enhanced_query, chunked_docs, embedder)
            else:
                with st.spinner("Drafting email directly..."):
                    # Generate Email without context
                    result = generate_email_direct(enhanced_query)
                
            # Display the result
            st.header("3. Generated Email")
            st.write(result["email"])
            
            # Display citations
            if result["citations"]:
                st.caption(f"**Sources used:** {', '.join(result['citations'])}")
                
            # Download button for the generated text
            st.download_button(
                label="📥 Download Email as TXT",
                data=result["email"],
                file_name="generated_email.txt",
                mime="text/plain"
            )
            
            logger.info("Email generated and displayed successfully.")
            
        except Exception as e:
            # Catch any unexpected errors and display them cleanly to the user
            st.error(f"An error occurred during processing: {str(e)}")
            logger.error(f"App processing error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

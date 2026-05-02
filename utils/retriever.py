import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import config

# Configure logging for the retriever module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_email(query: str, chunked_docs: List[Document], embedder: Any) -> Dict[str, Any]:
    """
    Creates a FAISS vector store, retrieves relevant context, and generates an email using Gemini.
    
    Args:
        query (str): The enhanced user query.
        chunked_docs (List[Document]): The chunked context documents.
        embedder (Any): The HuggingFace embedding model.
        
    Returns:
        Dict[str, Any]: A dictionary containing the generated email text and the sources used.
        
    Example:
        >>> result = generate_email("Draft an email about X", chunks, embedder)
        >>> print(result["email"])
    """
    try:
        # Step 1: Create a local FAISS vector store from the document chunks
        # This allows for fast similarity search based on the query
        logger.info("Building FAISS vector store...")
        vectorstore = FAISS.from_documents(chunked_docs, embedder)
        
        # Step 2: Set up the retriever
        # We retrieve the top 4 most relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Step 3: Initialize the Gemini LLM
        logger.info(f"Initializing LLM: {config.LLM_MODEL_NAME}")
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME, 
            temperature=config.TEMPERATURE
        )
        
        # Step 4: Define the prompt template for email generation
        # We explicitly ask the model to act as a professional email writer
        system_prompt = (
            "You are an expert AI Email Writer. Your task is to write a highly professional, "
            "clear, and concise email based on the user's request and the provided context.\n"
            "If the context does not contain relevant information, state that you cannot fulfill the request "
            "based on the provided documents. Do NOT make up information.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Step 5: Create the retrieval chain
        # This chain first fetches documents and then passes them to the LLM
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Step 6: Invoke the chain with the user's query
        logger.info(f"Invoking RAG chain for query: {query}")
        response = rag_chain.invoke({"input": query})
        
        # Extract the final answer and the source documents for citations
        email_content = response.get("answer", "")
        
        # New SDK versions sometimes return a list of content blocks instead of a flat string
        if isinstance(email_content, list):
            extracted_text = ""
            for item in email_content:
                if isinstance(item, dict) and "text" in item:
                    extracted_text += item["text"]
                elif isinstance(item, str):
                    extracted_text += item
            email_content = extracted_text
        elif not isinstance(email_content, str):
            email_content = str(email_content)
            
        source_docs = response.get("context", [])
        
        # Extract unique source file names to provide as citations
        citations = list(set([doc.metadata.get("source", "Unknown Source") for doc in source_docs]))
        
        logger.info("Successfully generated email.")
        return {
            "email": email_content,
            "citations": citations
        }
        
    except Exception as e:
        logger.error(f"Error during email generation: {e}")
        raise RuntimeError(f"Failed to generate email: {str(e)}")

def generate_email_direct(query: str) -> Dict[str, Any]:
    """
    Generates an email directly using Gemini without any retrieved context.
    
    Args:
        query (str): The user query.
        
    Returns:
        Dict[str, Any]: A dictionary containing the generated email text and empty citations.
    """
    try:
        logger.info(f"Initializing LLM for direct generation: {config.LLM_MODEL_NAME}")
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME, 
            temperature=config.TEMPERATURE
        )
        
        system_prompt = (
            "You are an expert AI Email Writer. Your task is to write a highly professional, "
            "clear, and concise email based on the user's request."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        
        logger.info(f"Invoking direct LLM chain for query: {query}")
        response = chain.invoke({"input": query})
        
        email_content = response.content
        
        # Handle list format from newer Gemini SDKs
        if isinstance(email_content, list):
            extracted_text = ""
            for item in email_content:
                if isinstance(item, dict) and "text" in item:
                    extracted_text += item["text"]
                elif isinstance(item, str):
                    extracted_text += item
            email_content = extracted_text
        elif not isinstance(email_content, str):
            email_content = str(email_content)
            
        logger.info("Successfully generated email directly.")
        return {
            "email": email_content,
            "citations": []
        }
        
    except Exception as e:
        logger.error(f"Error during direct email generation: {e}")
        raise RuntimeError(f"Failed to generate email directly: {str(e)}")

# AI Email Writer

A completely automated, LangChain-powered RAG (Retrieval-Augmented Generation) application to draft professional emails based on uploaded context documents.

## Model Architecture

The application is built on a robust RAG (Retrieval-Augmented Generation) architecture utilizing the following components:
- **Language Model**: Google Gemini Flash (`gemini-flash-latest`) for high-speed, high-quality text generation and reasoning.
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` for generating dense vector representations of text chunks locally (CPU-optimized).
- **Vector Store**: FAISS (Facebook AI Similarity Search) for efficient, local similarity search and retrieval of relevant document chunks.
- **Orchestration**: LangChain, managing the document loading, text splitting (chunking), vector storage, and the retrieval QA pipeline.
- **NLP Processing**: spaCy (`en_core_web_sm`) utilized for input validation, entity extraction, and query enhancement.
- **Frontend**: Streamlit, providing an interactive, web-based UI for document uploading and query submission.

## Run Guide

Follow these exact steps to run the application:

**Step 1: Create a virtual environment**
```bash
python -m venv venv
```

**Step 2: Activate the virtual environment**
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Step 4: Configure environment variables**
```bash
cp .env.example .env
```
Open `.env` and replace `your_gemini_api_key_here` with your actual Google Gemini API key.

**Step 5: Run Validation Tests**
```bash
python -m pytest test_validation.py
```
*Expected Output: All 5 tests pass (green dots).*

**Step 6: Run the Application**
```bash
streamlit run app.py
```

**Step 7: Usage**
1. Open `http://localhost:8501` in your browser.
2. Upload a sample document (PDF/TXT) via the sidebar.
3. Enter your query (e.g., "Draft a status update email").
4. Click "Generate Email" and verify the output and citations. Download the result if needed.

## Troubleshooting (Top Fixes)

If a step fails, here is the exact cause and fix:
1. **ModuleNotFoundError**: Cause: Virtual environment not activated or packages not installed. Fix: Ensure `venv\Scripts\activate` is run, then `pip install -r requirements.txt`.
2. **Missing Gemini API Key Error**: Cause: `.env` file is missing or `GEMINI_API_KEY` is not set. Fix: Complete Step 4 above and restart the app.
3. **spaCy model not found**: Cause: Did not download the language model. Fix: Run `python -m spacy download en_core_web_sm`.
4. **FAISS index error / No CPU**: Cause: Incompatible architecture. Fix: Ensure `faiss-cpu` is installed, not `faiss-gpu` unless you have an NVIDIA GPU configured.
5. **Streamlit Port In Use**: Cause: Another instance of Streamlit is running. Fix: Stop the old instance (Ctrl+C) or run `streamlit run app.py --server.port 8502`.

## Validation Checklist

- [x] `python -m pytest test_validation.py` → all 5 PASS
- [x] `streamlit run app.py` → opens localhost:8501, no errors
- [x] `python -c "import config; print(config.check_keys())"` → Prints `True` (if key is set)
- [x] Upload sample file → output displayed, no crash
- [x] Submit empty input → `st.error()` shown, no crash
- [x] Submit valid query → non-empty answer with citations
- [x] Edge cases handled gracefully (too short queries, etc.)

import sys
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import hashlib
import pickle
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn
from typing import Optional
import tempfile
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Configure Windows console for UTF-8 if needed
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')


# Configure logging
def configure_logging():
    """Configure logging with proper encoding and handlers"""
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler with UTF-8 encoding
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


configure_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PDF Question Answering System",
    description="API for querying PDF documents using semantic search and Gemini AI",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
class Config:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_CHUNKS = 5
    MAX_CONTEXT_LENGTH = 10000  # characters
    TEMP_DIR = "temp_files"
    INDEX_DIR = "indexes"
    GEMINI_MODEL = "gemini-1.5-flash"
    MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB


# Create necessary directories
os.makedirs(Config.TEMP_DIR, exist_ok=True)
os.makedirs(Config.INDEX_DIR, exist_ok=True)


# Load models and services
def initialize_services():
    """Initialize all required services with proper error handling"""
    try:
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully")

        return embedding_model
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise


model = initialize_services()


# Utility functions
def hash_pdf(file_bytes: bytes) -> str:
    """Generate a consistent hash for PDF files"""
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()
    logger.info(f"Generated PDF hash: {pdf_hash}")
    return pdf_hash


def save_index_and_chunks(pdf_hash: str, index: faiss.Index, chunks: list):
    """Save FAISS index and text chunks with error handling"""
    try:
        index_file = os.path.join(Config.INDEX_DIR, f"{pdf_hash}.index")
        chunks_file = os.path.join(Config.INDEX_DIR, f"{pdf_hash}_chunks.pkl")

        faiss.write_index(index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)

        logger.info(f"Saved index and chunks for hash {pdf_hash}")
    except Exception as e:
        logger.error(f"Failed to save index/chunks: {str(e)}")
        raise


def load_index_and_chunks(pdf_hash: str) -> tuple:
    """Load FAISS index and text chunks with error handling"""
    try:
        index_file = os.path.join(Config.INDEX_DIR, f"{pdf_hash}.index")
        chunks_file = os.path.join(Config.INDEX_DIR, f"{pdf_hash}_chunks.pkl")

        if not (os.path.exists(index_file) and os.path.exists(chunks_file)):
            raise FileNotFoundError("Index or chunks file not found")

        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        logger.info(f"Loaded cached index and chunks for hash {pdf_hash}")
        return index, chunks
    except Exception as e:
        logger.error(f"Failed to load index/chunks: {str(e)}")
        raise


def chunk_text(text: str, chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks with improved handling"""
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap

    logger.info(f"Created {len(chunks)} text chunks")
    return chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF with improved error handling"""
    try:
        with tempfile.NamedTemporaryFile(dir=Config.TEMP_DIR, suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        full_text = ""
        try:
            reader = PdfReader(temp_file_path)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                    logger.debug(f"Processed page {i + 1}")
                except Exception as page_error:
                    logger.warning(f"Error processing page {i + 1}: {str(page_error)}")
                    continue
        finally:
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temp file: {str(cleanup_error)}")

        if not full_text.strip():
            raise ValueError("No text could be extracted from PDF")

        return full_text
    except Exception as e:
        logger.error(f"PDF text extraction failed: {str(e)}")
        raise


def generate_context_prompt(context: str, question: str) -> str:
    """Generate optimized prompt for Gemini"""
    if len(context) > Config.MAX_CONTEXT_LENGTH:
        context = context[:Config.MAX_CONTEXT_LENGTH] + "\n\n[CONTEXT TRUNCATED]"

    return f"""
**Document Context:**
{context}

**Question:**
{question}

**Instructions:**
1. Answer the question using ONLY the provided document context.
2. Be precise and concise.
3. If the answer isn't in the document, say "The document doesn't contain information to answer this question."
4. Use markdown formatting for clarity (bullet points, bold, etc. when helpful).
5. Include relevant excerpts from the document when appropriate.

**Answer:**
"""


@app.post("/ask")
async def ask_pdf(
        file: UploadFile = File(..., description="PDF file to query"),
        question: str = Form(..., description="Question about the PDF content"),
        top_k: Optional[int] = Form(Config.TOP_K_CHUNKS, description="Number of chunks to retrieve")
):
    """Endpoint for querying PDF documents"""
    start_time = datetime.now()
    logger.info(f"New request - File: {file.filename}, Question: {question}")

    try:
        # Validate inputs
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Read and validate file size
        contents = await file.read()
        if len(contents) > Config.MAX_PDF_SIZE:
            raise HTTPException(status_code=413,
                              detail=f"PDF file too large. Max size is {Config.MAX_PDF_SIZE // (1024 * 1024)}MB")

        pdf_hash = hash_pdf(contents)

        # Check for cached index
        index_path = os.path.join(Config.INDEX_DIR, f"{pdf_hash}.index")
        chunks_path = os.path.join(Config.INDEX_DIR, f"{pdf_hash}_chunks.pkl")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            logger.info("Using cached index")
            index, chunks = load_index_and_chunks(pdf_hash)
        else:
            logger.info("Building new index")
            full_text = extract_text_from_pdf(contents)
            chunks = chunk_text(full_text)

            if not chunks:
                raise HTTPException(status_code=400, detail="No meaningful text could be extracted from PDF")

            logger.info("Generating embeddings...")
            embeddings = model.encode(chunks).astype("float32")

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            save_index_and_chunks(pdf_hash, index, chunks)

        # Process question
        logger.info("Processing question...")
        query_embedding = model.encode([question]).astype("float32")
        _, indices = index.search(query_embedding, top_k)

        retrieved_chunks = []
        for i in indices[0]:
            if 0 <= i < len(chunks):
                retrieved_chunks.append(chunks[i])

        if not retrieved_chunks:
            raise HTTPException(status_code=400, detail="No relevant content found in document")

        context = "\n\n---\n\n".join(retrieved_chunks)

        # Generate response with Gemini
        logger.info("Querying Gemini...")
        prompt = generate_context_prompt(context, question)

        try:
            gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
            response = gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1000
                }
            )

            if not response.text:
                raise HTTPException(status_code=500, detail="Empty response from Gemini")

            answer = response.text.strip()
            logger.info("Successfully generated answer")

            return {
                "answer": answer,
                "context_chunks": retrieved_chunks,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "document_hash": pdf_hash
            }

        except Exception as gemini_error:
            logger.error(f"Gemini error: {str(gemini_error)}")
            raise HTTPException(status_code=500, detail="Error generating answer from AI model")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the request.")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "embedding_model": "active",
            "gemini": "active" if os.getenv("GOOGLE_API_KEY") else "inactive"
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_config=None,
        access_log=False
    )
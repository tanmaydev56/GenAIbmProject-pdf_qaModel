from fastapi import FastAPI, File, UploadFile, Form
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

load_dotenv()
app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and setup once
model = SentenceTransformer('all-MiniLM-L6-v2')
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
def hash_pdf(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()
def save_index_and_chunks(pdf_hash, index, chunks):
    faiss.write_index(index, f"indexes/{pdf_hash}.index")
    with open(f"indexes/{pdf_hash}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_chunks(pdf_hash):
    index = faiss.read_index(f"indexes/{pdf_hash}.index")
    with open(f"indexes/{pdf_hash}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

@app.post("/ask")
async def ask_pdf(file: UploadFile = File(...), question: str = Form(...)):
    contents = await file.read()
    pdf_hash = hash_pdf(contents)

    index_path = f"indexes/{pdf_hash}.index"
    chunks_path = f"indexes/{pdf_hash}_chunks.pkl"

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index, chunks = load_index_and_chunks(pdf_hash)
        print("✅ Reusing existing FAISS index.")
    else:
        print("⚠️ New PDF. Building FAISS index...")
        with open("temp.pdf", "wb") as f:
            f.write(contents)

        reader = PdfReader("temp.pdf")
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        def chunk_text(text, chunk_size=1000, overlap=200):
            chunks = []
            start = 0
            while start < len(text):
                chunk = text[start:start + chunk_size]
                chunks.append(chunk)
                start += chunk_size - overlap
            return chunks

        chunks = chunk_text(full_text)

        embeddings = model.encode(chunks).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        os.makedirs("indexes", exist_ok=True)
        save_index_and_chunks(pdf_hash, index, chunks)

    # Embed user question
    query_embedding = model.encode([question]).astype("float32")
    _, I = index.search(query_embedding, 3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Prepare prompt and query Gemini
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are a highly intelligent AI assistant specialized in question answering. You are provided with extracted content from a PDF document. 

Your task is to answer the user's question as accurately, clearly, and concisely as possible using only the information contained in the given context. Do not assume or fabricate any information that is not explicitly stated in the context. 

If the answer cannot be found within the context, respond with: 
"The provided document does not contain enough information to answer that question."

Use markdown formatting in your answer if it helps clarity.

Context:
{context}

Question:
{question}

Answer:
"""

    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    response = gemini_model.generate_content(prompt)

    return {"answer": response.text}
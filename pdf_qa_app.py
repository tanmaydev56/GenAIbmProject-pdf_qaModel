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
import os


# ---------------------- Configuration ----------------------
API_KEY =os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

model = SentenceTransformer("all-MiniLM-L6-v2")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Utility Functions ----------------------
def hash_pdf(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def save_index_and_chunks(pdf_hash, index, chunks):
    os.makedirs("indexes", exist_ok=True)
    faiss.write_index(index, f"indexes/{pdf_hash}.index")
    with open(f"indexes/{pdf_hash}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_chunks(pdf_hash):
    index = faiss.read_index(f"indexes/{pdf_hash}.index")
    with open(f"indexes/{pdf_hash}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_text_from_pdf(pdf_bytes):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)
    reader = PdfReader("temp.pdf")
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def query_gemini(context, question):
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
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ---------------------- FastAPI Endpoint ----------------------
@app.post("/ask")
async def ask_pdf(file: UploadFile = File(...), question: str = Form(...)):
    contents = await file.read()
    pdf_hash = hash_pdf(contents)

    index_path = f"indexes/{pdf_hash}.index"
    chunks_path = f"indexes/{pdf_hash}_chunks.pkl"

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index, chunks = load_index_and_chunks(pdf_hash)
    else:
        text = extract_text_from_pdf(contents)
        chunks = chunk_text(text)
        embeddings = model.encode(chunks).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        save_index_and_chunks(pdf_hash, index, chunks)

    query_embedding = model.encode([question]).astype("float32")
    _, I = index.search(query_embedding, 3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    answer = query_gemini("\n\n".join(retrieved_chunks), question)
    return {"answer": answer}

# ---------------------- CLI Mode (Optional) ----------------------
if __name__ == "__main__":
    pdf_path = input("Enter path to PDF file: ").strip()
    question = input("Ask a question about the PDF: ").strip()

    with open(pdf_path, "rb") as f:
        contents = f.read()

    pdf_hash = hash_pdf(contents)
    index_path = f"indexes/{pdf_hash}.index"
    chunks_path = f"indexes/{pdf_hash}_chunks.pkl"

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index, chunks = load_index_and_chunks(pdf_hash)
        print("✅ Using cached FAISS index.")
    else:
        text = extract_text_from_pdf(contents)
        chunks = chunk_text(text)
        embeddings = model.encode(chunks).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        save_index_and_chunks(pdf_hash, index, chunks)
        print("✅ New index built and saved.")

    query_embedding = model.encode([question]).astype("float32")
    _, I = index.search(query_embedding, 3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    print("\n📄 Retrieved chunks:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"--- Chunk {i} ---\n{chunk}\n")

    answer = query_gemini("\n\n".join(retrieved_chunks), question)
    print("\n🤖 Gemini's Answer:\n")
    print(answer)

port = int(os.environ.get("PORT", 8000))  # fallback to 8000 for local
uvicorn.run(app, host="0.0.0.0", port=port)
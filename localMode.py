import os
import numpy as np
import faiss
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ✅ Step 1: Load and extract text from PDF
pdf_path = "C:/Users/TANMAY SHARMA/Desktop/10RBSEResults.pdf"
reader = PdfReader(pdf_path)

full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print("✅ PDF uploaded and text extracted.")
print("First 1000 characters of text:\n")
print(full_text[:1000])

# ✅ Step 2: Split text into overlapping chunks
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)
print(f"✅ Split into {len(chunks)} chunks.")
print("Preview of one chunk:\n")
print(chunks[0])

# ✅ Step 3: Generate embeddings for chunks
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# ✅ Step 4: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"✅ FAISS index built with {len(chunks)} chunks.")

# ✅ Step 5: User query input
question = input("🔍 Ask a question about your PDF: ")

# Generate query embedding
query_embedding = embedding_model.encode([question]).astype("float32")

# Search for top-k similar chunks
k = 3
D, I = index.search(query_embedding, k)
retrieved_chunks = [chunks[i] for i in I[0]]

print(f"\n✅ Top {k} relevant chunks:\n")
for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"--- Chunk {i} ---\n{chunk}\n")

# ✅ Step 6: Gemini AI Response
genai.configure(api_key="AIzaSyCfLXqdvUysY-V9sS38UtypW5DtCVQqS7U")  # Replace with your key

context = "\n\n".join(retrieved_chunks)
final_prompt = f"""
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

# Generate response from Gemini
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
response = gemini_model.generate_content(final_prompt)

print("\n🤖 Gemini's Answer:\n")
print(response.text.strip())
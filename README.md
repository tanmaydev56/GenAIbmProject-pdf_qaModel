# 📄 Gemini PDF Q&A App

A powerful PDF-based Question Answering app using Google's gemini-1.5-flash, FAISS, and SentenceTransformers. Ask natural language questions and get accurate answers from any uploaded PDF.

---

## 🚀 Features

- 🔍 Ask questions about your PDF using natural language.
- 🧠 Embeds PDF content using `sentence-transformers` for semantic search.
- ⚡ Fast vector search powered by FAISS.
- ⚡ indexes are created and stored inside the root directory, So that if the same file is uploaded again then the model will use the previos generated index file will save a lot of time. 
- 🤖 Smart answers generated using Gemini flash.
- 📚 Clean chunking and preprocessing for high-quality context retrieval.

---

## 🧰 Tech Stack

| Tool / Library         | Purpose                                |
|------------------------|----------------------------------------|
| NextJS                 | Frontend UI for interaction            |
| Gemini flash (Google)  | LLM to generate answers                |
| FAISS                  | Vector index for fast semantic search  |
| SentenceTransformers   | Embeddings from PDF & question text    |
| PyMuPDF (fitz)         | PDF parsing and text extraction        |
| NumPy, re, os          | Supporting utilities                   |

---

## 📦 Installation

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/pdf-qa-gemini.git
   cd pdf-qa-gemini
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Gemini API Key**

   Add your API key as an environment variable:
   ```bash
   export GEMINI_API_KEY='your_api_key_here'
   ```

---

## 🧠 How It Works

1. **PDF Upload:** You upload a PDF file via the Next js UI.
2. **Text Extraction:** Text is extracted using PyMuPDF (`fitz`).
3. **Chunking:** Text is split into smaller chunks for better embedding and retrieval( the relevant top_k_chunks are created according to the user question about the pdf.
4. **Embedding:** Each chunk is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`).
5. **FAISS Indexing:** Chunks are stored in a FAISS index.
6. **QA Process:**
   - The user's question is embedded.
   - The top-K most relevant chunks are retrieved from FAISS.
   - These chunks + the question are sent to Gemini.
   - Gemini responds with a concise answer.
7. **Answer Display:** The answer is shown in the NextJS user interface.

---

## 🧪 Example Usage
upload:
> "user uploads the pdf"

Ask:
> "What is the conclusion of this report?"

Get:
> "The report concludes that renewable energy adoption must be accelerated by 2030 to meet global emission targets..."

---

## ⚙️ Configuration

You can tweak the number of retrieved chunks for context:

```python
top_k_chunks = 5  # Try values like 3, 5, or 7
```

This controls how much context Gemini gets when answering a question.

---

## 📂 Project Structure

```
pdf-qa-gemini/
│
├── app.py              # Main Streamlit application
├── indexes/
    ├──40....
    .....
├── requirements.txt    # Python dependencies
├── utils/
│   ├── pdf_utils.py    # PDF text extraction
│   ├── faiss_utils.py  # FAISS indexing and searching
│   └── gemini_utils.py # Gemini prompt & response
```

---

## ✅ To Do

- [ ] Add support for multiple PDFs
- [ ] Add OpenAI or Claude as alternative LLMs
- [ ] Highlight answer location in original PDF
- [ ] Improve UI with file previews

---

## 💡 Credits

Created by [Tanmy Sharma] — Powered by Gemini, FAISS, and NextJs.

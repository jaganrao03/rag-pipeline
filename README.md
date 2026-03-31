# RAG Pipeline: PDF Question-Answering API

> Ask natural language questions about any PDF document.
> Built with **LangChain**, **Claude (Anthropic)**, **Pinecone**, and **FastAPI**.

---

## What This Does

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline — a pattern used in production AI systems to give an LLM access to your specific documents, rather than relying only on its training data.

You point it at a PDF, ingest it once, then ask questions via a REST API. The system finds the most relevant sections of the document and uses Claude to answer in plain English, with citations back to the source pages.

---

## Architecture

```
                         INGESTION (run once)
 ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
 │  PDF     │───▶│  PyPDF       │───▶│ Text Splitter   │───▶│  Pinecone    │
 │ Document │    │  (load pages)│    │ (1000-char      │    │  Embeddings  │
 └──────────┘    └──────────────┘    │  chunks, 200    │    │  (E5 model)  │
                                     │  overlap)       │    └──────┬───────┘
                                     └─────────────────┘           │
                                                                    ▼
                                                             ┌──────────────┐
                                                             │   Pinecone   │
                                                             │  Vector DB   │
                                                             │  (1024-dim)  │
                                                             └──────┬───────┘
                                                                    │
                         QUERY (on each question)                   │
 ┌──────────┐    ┌──────────────┐    ┌─────────────────┐           │
 │  Answer  │◀───│   Claude     │◀───│ Top-K Chunks    │◀──────────┘
 │  + Sources│   │  (Sonnet)    │    │ (similarity     │
 └──────────┘    └──────────────┘    │  search)        │
                                     └─────────────────┘
```

### Key Concepts

- **Chunking**: The PDF is split into ~1000-character overlapping pieces so each chunk is small enough to be meaningful on its own.
- **Embeddings**: Each chunk is converted to a 1024-number vector that captures its meaning. Similar text produces similar vectors.
- **Vector Search**: When you ask a question, Pinecone finds the chunks whose vectors are closest to your question's vector — semantic search, not keyword matching.
- **Grounded Generation**: Claude only uses the retrieved chunks to answer. This prevents hallucination and keeps answers traceable to the source.

---

## Tech Stack

| Component    | Tool                          | Why                                      |
|--------------|-------------------------------|------------------------------------------|
| LLM          | Claude (Anthropic)            | Best reasoning quality, generous context |
| Vector DB    | Pinecone                      | Fully managed, generous free tier        |
| Embeddings   | Pinecone E5 (multilingual)    | Free, no extra API key, 1024-dim         |
| Orchestration| LangChain                     | Standard RAG tooling, clean abstractions |
| API Framework| FastAPI                       | Auto-docs at `/docs`, fast, typed        |
| PDF Parsing  | PyPDF                         | Pure Python, no system dependencies      |

---

## Prerequisites

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/) (free tier available)
- [Pinecone API key](https://app.pinecone.io/) (free tier available)

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your ANTHROPIC_API_KEY and PINECONE_API_KEY
```

### 3. Create a Pinecone index

In the [Pinecone console](https://app.pinecone.io/):

1. Click **Create Index**
2. Set **Index Name**: `rag-pipeline`
3. Set **Dimensions**: `1024`
4. Set **Metric**: `Cosine`
5. Select **Serverless** → **AWS us-east-1** (free tier)
6. Click **Create Index**

> **Why 1024?** The `multilingual-e5-large` embedding model produces 1024-dimensional vectors. The index dimensions must match exactly.

### 4. Get a sample PDF

Download any public technical PDF. For example, the "Attention Is All You Need" paper:

```bash
# Linux/Mac
curl -L https://arxiv.org/pdf/1706.03762 -o docs/sample.pdf
```

Or drop any PDF you want into the `docs/` folder and rename it `sample.pdf`.

### 5. Ingest the document

```bash
python scripts/run_ingest.py docs/sample.pdf
```

You should see output like:
```
=== Starting ingestion: docs/sample.pdf ===
Loading PDF: docs/sample.pdf
  Loaded 15 pages
  Split into 87 chunks (size=1000, overlap=200)
Connecting to Pinecone index 'rag-pipeline'...
  Index 'rag-pipeline' already exists — skipping creation.
  Embedding and uploading 87 chunks (this may take a minute)...
  Upload complete.
=== Ingestion complete: 87 chunks stored in Pinecone ===
```

### 6. Start the API

```bash
uvicorn src.api:app --reload
```

### 7. Ask questions

```bash
# Check the API is running
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention and why is it useful?"}'
```

Or open **http://localhost:8000/docs** in a browser for an interactive UI.

---

## API Reference

### `GET /health`

Confirms the API is running.

```json
{"status": "ok", "message": "RAG Pipeline API is running."}
```

### `POST /ingest`

Ingests a PDF into Pinecone.

**Request:**
```json
{"file_path": "docs/sample.pdf"}
```

**Response:**
```json
{
  "message": "Successfully ingested 'docs/sample.pdf'",
  "chunks_ingested": 87
}
```

### `POST /ask`

Answers a question using the ingested document.

**Request:**
```json
{"question": "What is multi-head attention?"}
```

**Response:**
```json
{
  "question": "What is multi-head attention?",
  "answer": "Multi-head attention is a mechanism that allows the model to jointly attend to information from different representation subspaces at different positions...",
  "sources": [
    {
      "content": "Multi-head attention allows the model to jointly attend to information...",
      "page": 4,
      "source": "docs/sample.pdf"
    }
  ]
}
```

---

## Project Structure

```
rag-pipeline/
├── .env.example          # Template — copy to .env and fill in keys
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   └── sample.pdf        # Your PDF goes here (gitignored for large files)
├── src/
│   ├── __init__.py
│   ├── config.py         # Settings loaded from .env
│   ├── ingest.py         # PDF → chunks → Pinecone
│   ├── query.py          # Question → Pinecone search → Claude answer
│   └── api.py            # FastAPI endpoints
├── scripts/
│   └── run_ingest.py     # Run this once to ingest a PDF
└── tests/
    └── test_query.py     # Smoke tests (pytest)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## How It Works — Step by Step

### Ingestion Pipeline

1. **Load**: `PyPDFLoader` reads the PDF page by page
2. **Split**: `RecursiveCharacterTextSplitter` breaks pages into ~1000-character chunks with 200-character overlap (so context isn't lost at chunk boundaries)
3. **Embed**: Each chunk is converted to a 1024-dimension vector using Pinecone's hosted `multilingual-e5-large` model
4. **Store**: Vectors + original text + metadata (page number, source) are upserted into Pinecone

### Query Pipeline

1. **Embed query**: The question is converted to a vector using the same embedding model
2. **Search**: Pinecone performs approximate nearest-neighbor search and returns the top-5 most similar chunks
3. **Prompt**: A system prompt is assembled with the retrieved chunks as context, instructing Claude to answer only from this context
4. **Generate**: Claude returns a grounded answer
5. **Return**: The API returns the answer + source citations so the response is verifiable

---

## Configuration

All settings are in `src/config.py`:

| Variable         | Default                   | Description                              |
|------------------|---------------------------|------------------------------------------|
| `CHUNK_SIZE`     | `1000`                    | Characters per chunk                     |
| `CHUNK_OVERLAP`  | `200`                     | Overlap between chunks                   |
| `TOP_K`          | `5`                       | Number of chunks retrieved per question  |
| `EMBEDDING_MODEL`| `multilingual-e5-large`   | Pinecone's hosted embedding model        |
| `CLAUDE_MODEL`   | `claude-opus-4-5`         | Claude model for answer generation       |

---

## Cost Estimates (Free Tier)

- **Pinecone**: Free tier includes 2GB storage (~1M vectors) — sufficient for hundreds of PDFs
- **Anthropic**: Claude API charges per token. A single question with 5 retrieved chunks costs roughly $0.001–$0.005

---

## Future Improvements

- [ ] Streaming responses (Server-Sent Events)
- [ ] Support multiple PDFs with source filtering
- [ ] Streamlit or React frontend
- [ ] Deploy to GCP Cloud Run
- [ ] Conversation memory (multi-turn Q&A)
- [ ] Hybrid search (semantic + keyword)

---

## Author

**Jagan** — BI Developer / Technology Consulting Analyst  
Building toward AI Engineering.  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

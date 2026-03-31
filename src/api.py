"""
api.py — FastAPI web server that exposes the RAG pipeline as REST endpoints.

Once running, you can:
  - Open http://localhost:8000/docs in a browser for interactive API documentation
  - POST to /ask with a JSON body to ask questions
  - POST to /ingest with a file path to add a new document

Start the server with:
  uvicorn src.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ingest import run_ingestion
from src.query import ask_question

# Create the FastAPI application
app = FastAPI(
    title="RAG Pipeline API",
    description="Ask natural language questions about PDF documents using Claude + Pinecone.",
    version="1.0.0",
)


# ── Request / Response Models ─────────────────────────────────────────────────
# Pydantic models define the shape of JSON that the API accepts and returns.
# FastAPI automatically validates incoming requests against these models.

class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""
    file_path: str  # Path to the PDF file, e.g. "docs/sample.pdf"


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint."""
    message: str
    chunks_ingested: int


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str  # Natural language question, e.g. "What is multi-head attention?"


class SourceDocument(BaseModel):
    """A single source chunk returned alongside the answer."""
    content: str   # The text of the chunk (first 300 chars)
    page: int | None   # Page number in the original PDF
    source: str    # File path of the source document


class AskResponse(BaseModel):
    """Response from the /ask endpoint."""
    question: str
    answer: str
    sources: list[SourceDocument]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Quick check to confirm the API is running.
    Use this to verify deployment is working before running queries.
    """
    return {"status": "ok", "message": "RAG Pipeline API is running."}


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    """
    Ingests a PDF file into Pinecone.

    - Reads the PDF at the given file_path
    - Splits it into chunks
    - Embeds and stores chunks in Pinecone

    Only needs to be run once per document. After ingestion,
    you can ask questions without re-ingesting.
    """
    try:
        chunks_count = run_ingestion(request.file_path)
        return IngestResponse(
            message=f"Successfully ingested '{request.file_path}'",
            chunks_ingested=chunks_count,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {request.file_path}. Make sure the path is correct.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Answers a natural language question using the ingested document.

    - Embeds the question
    - Searches Pinecone for the most relevant document chunks
    - Sends the question + context to Claude
    - Returns the answer with source citations

    Make sure you've ingested a document first (POST /ingest).
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = ask_question(request.question)
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}",
        )

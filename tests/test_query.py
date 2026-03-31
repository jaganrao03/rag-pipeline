"""
test_query.py — Basic smoke tests for the RAG Pipeline.

Run with: pytest tests/
These tests check that the code structure and models work correctly
WITHOUT making real API calls (to avoid costs during testing).
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


# ── API Model Tests ────────────────────────────────────────────────────────────

def test_ask_request_model():
    """AskRequest should accept a question string."""
    from src.api import AskRequest
    req = AskRequest(question="What is a transformer?")
    assert req.question == "What is a transformer?"


def test_ask_request_empty_string():
    """AskRequest should accept any string (validation happens in the endpoint)."""
    from src.api import AskRequest
    req = AskRequest(question="")
    assert req.question == ""


def test_ingest_request_model():
    """IngestRequest should accept a file path string."""
    from src.api import IngestRequest
    req = IngestRequest(file_path="docs/sample.pdf")
    assert req.file_path == "docs/sample.pdf"


def test_source_document_model():
    """SourceDocument should correctly store chunk data."""
    from src.api import SourceDocument
    src = SourceDocument(content="Some text", page=3, source="docs/sample.pdf")
    assert src.page == 3
    assert src.source == "docs/sample.pdf"


# ── Config Tests ───────────────────────────────────────────────────────────────

def test_config_chunk_size():
    """Chunk size should be a positive integer."""
    # We need to mock env vars so config.py doesn't fail on missing keys
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-key",
        "PINECONE_API_KEY": "test-key",
        "PINECONE_INDEX_NAME": "test-index",
    }):
        from src import config
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
        assert config.TOP_K > 0


# ── API Endpoint Tests ─────────────────────────────────────────────────────────

def test_health_endpoint():
    """The /health endpoint should always return 200 OK."""
    # Patch env vars to prevent config errors during import
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-key",
        "PINECONE_API_KEY": "test-key",
        "PINECONE_INDEX_NAME": "test-index",
    }):
        from src.api import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_ask_endpoint_empty_question():
    """The /ask endpoint should return 400 for empty questions."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-key",
        "PINECONE_API_KEY": "test-key",
        "PINECONE_INDEX_NAME": "test-index",
    }):
        from src.api import app
        client = TestClient(app)
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 400


def test_ingest_endpoint_file_not_found():
    """The /ingest endpoint should return 404 for missing files."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-key",
        "PINECONE_API_KEY": "test-key",
        "PINECONE_INDEX_NAME": "test-index",
    }):
        from src.api import app
        client = TestClient(app)
        response = client.post("/ingest", json={"file_path": "docs/does_not_exist.pdf"})
        assert response.status_code == 404

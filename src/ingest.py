"""
ingest.py — Reads a PDF file, splits it into chunks, and stores them in Pinecone.

Run this ONCE per document (or whenever you want to update the document).
After ingestion, you can ask questions via the API without re-ingesting.

Pipeline:
  PDF file → pages → text chunks → Pinecone vector store
"""

import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_pdf(file_path: str) -> list:
    """
    Reads a PDF and returns a list of LangChain Document objects.
    Each Document contains the text of one page plus metadata (page number, source).
    """
    print(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")
    return documents


def chunk_documents(documents: list) -> list:
    """
    Splits each page into smaller overlapping chunks.

    Why chunk? Large language models have a context limit. Storing the entire
    document as one unit would make retrieval useless — we need granular pieces
    so the search can find the most relevant paragraph, not just the right page.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentence breaks, then words — keeping chunks semantically whole.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # max characters per chunk
        chunk_overlap=CHUNK_OVERLAP, # overlap so context isn't lost at boundaries
        length_function=len,         # measure length in characters (not tokens)
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def ensure_index_exists(pc: Pinecone) -> None:
    """
    Creates the Pinecone index if it doesn't already exist.
    Waits until the index is ready before returning.
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"  Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for the index to be ready (usually 10-30 seconds)
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(5)
        print(f"  Index '{PINECONE_INDEX_NAME}' is ready.")
    else:
        print(f"  Index '{PINECONE_INDEX_NAME}' already exists — skipping creation.")


def store_in_pinecone(chunks: list) -> PineconeVectorStore:
    """
    Converts text chunks to embeddings and stores them in Pinecone.

    What are embeddings? Each chunk of text is converted to a list of 1024 numbers
    (a "vector") that captures its meaning. Chunks with similar meaning get similar
    vectors. Pinecone stores these vectors and can quickly find the most similar
    ones given a query vector.
    """
    print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")

    # Initialize Pinecone client and ensure the index exists
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index_exists(pc)

    # PineconeEmbeddings uses Pinecone's hosted embedding model — no extra API key needed
    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        pinecone_api_key=PINECONE_API_KEY,
    )

    print(f"  Embedding and uploading {len(chunks)} chunks (this may take a minute)...")

    # from_documents() embeds all chunks and upserts them into Pinecone in batches
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    print("  Upload complete.")
    return vector_store


def run_ingestion(file_path: str) -> int:
    """
    Main ingestion function. Loads a PDF, chunks it, and stores it in Pinecone.
    Returns the number of chunks ingested.
    """
    print(f"\n=== Starting ingestion: {file_path} ===\n")

    documents = load_pdf(file_path)
    chunks = chunk_documents(documents)
    store_in_pinecone(chunks)

    print(f"\n=== Ingestion complete: {len(chunks)} chunks stored in Pinecone ===\n")
    return len(chunks)

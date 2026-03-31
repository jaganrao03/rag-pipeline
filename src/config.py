"""
config.py — Loads all settings from the .env file.

Every other file in this project imports from here.
This way, if you ever change a setting, you only change it in ONE place.
"""

import os
from dotenv import load_dotenv

# load_dotenv() reads the .env file and puts each line into os.environ
load_dotenv()


def _require(var_name: str) -> str:
    """
    Gets an environment variable or raises a clear error if it's missing.
    Prevents cryptic failures deep in the code — fails fast at startup.
    """
    value = os.getenv(var_name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {var_name}\n"
            f"Did you copy .env.example to .env and fill in your keys?"
        )
    return value


# ── API Keys ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
PINECONE_API_KEY = _require("PINECONE_API_KEY")

# ── Pinecone Settings ────────────────────────────────────────────────────────
# The name of the index you created in the Pinecone console
PINECONE_INDEX_NAME = _require("PINECONE_INDEX_NAME")

# The embedding model hosted by Pinecone (free, no extra API key needed)
# "multilingual-e5-large" produces 1024-dimensional vectors
EMBEDDING_MODEL = "multilingual-e5-large"

# Your Pinecone index MUST be created with dimension=1024 to match this model
EMBEDDING_DIMENSION = 1024

# ── Claude Settings ──────────────────────────────────────────────────────────
# The Claude model to use for generating answers
CLAUDE_MODEL = "claude-opus-4-5"

# ── Chunking Settings ────────────────────────────────────────────────────────
# How many characters per text chunk (roughly 200-250 words)
CHUNK_SIZE = 1000

# How many characters overlap between consecutive chunks
# Overlap helps avoid cutting off sentences at chunk boundaries
CHUNK_OVERLAP = 200

# ── Retrieval Settings ───────────────────────────────────────────────────────
# How many document chunks to retrieve from Pinecone per question
# More chunks = more context for Claude, but also higher API cost
TOP_K = 5

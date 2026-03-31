"""
run_ingest.py — Command-line script to ingest a PDF into Pinecone.

Run this ONCE before you can ask questions:
  python scripts/run_ingest.py
  python scripts/run_ingest.py docs/my_document.pdf

Make sure you've set up .env with your API keys first.
"""

import sys
import os

# Add the project root to the Python path so "from src.xxx import" works
# (This is needed when running scripts/ directly rather than via uvicorn)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import run_ingestion

# Default to docs/sample.pdf if no argument is given
file_path = sys.argv[1] if len(sys.argv) > 1 else "docs/sample.pdf"

# Check the file exists before trying to ingest
if not os.path.exists(file_path):
    print(f"ERROR: File not found: {file_path}")
    print("Please provide a valid path to a PDF file.")
    print("Usage: python scripts/run_ingest.py path/to/your.pdf")
    sys.exit(1)

# Run the ingestion pipeline
count = run_ingestion(file_path)
print(f"Ready! You can now start the API and ask questions about '{file_path}'.")
print("Start the API with: uvicorn src.api:app --reload")

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Paths ──
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data" / "pdfs"
CHROMA_DIR     = BASE_DIR / "storage" / "chroma_db"
BM25_DIR       = BASE_DIR / "storage" / "bm25_index"
TESTS_DIR      = BASE_DIR / "tests"

# ── OpenAI ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL      = "gpt-4o-mini"
EMBED_MODEL    = "all-MiniLM-L6-v2"       # local, free, no API needed

# ── Chunking ──
CHUNK_SIZE     = 512    # tokens per child chunk
CHUNK_OVERLAP  = 50     # overlap between chunks
PARENT_SIZE    = 1024   # tokens per parent chunk

# ── Retrieval ──
BM25_TOP_K     = 20     # candidates from BM25
DENSE_TOP_K    = 20     # candidates from ChromaDB
RERANK_TOP_K   = 5      # final chunks after re-ranking

# ── Re-ranker ──
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── ChromaDB ──
COLLECTION_NAME = "pdf_chunks"
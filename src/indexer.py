# import json
# import pickle
# import sys
# from pathlib import Path
# from typing import List

# import chromadb
# from chromadb.utils import embedding_functions
# from rank_bm25 import BM25Okapi

# sys.path.append(str(Path(__file__).parent.parent))
# from config import CHROMA_DIR, BM25_DIR, COLLECTION_NAME, EMBED_MODEL
# from src.ingestor import Chunk, load_chunks, TESTS_DIR


# def get_chroma_client():
#     CHROMA_DIR.mkdir(parents=True, exist_ok=True)
#     return chromadb.PersistentClient(path=str(CHROMA_DIR))


# def get_or_create_collection(client):
#     ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name=EMBED_MODEL
#     )
#     return client.get_or_create_collection(
#         name=COLLECTION_NAME,
#         embedding_function=ef,
#         metadata={"hnsw:space": "cosine"}
#     )


# def index_to_chroma(collection, chunks: List[Chunk]):
#     child_chunks = [c for c in chunks if c.chunk_type == "child"]

#     existing = set(collection.get()["ids"])
#     new_chunks = [c for c in child_chunks if c.chunk_id not in existing]

#     if not new_chunks:
#         print(f"  ChromaDB: all {len(child_chunks)} chunks already indexed.")
#         return

#     # ChromaDB add in batches of 500
#     batch_size = 500
#     for i in range(0, len(new_chunks), batch_size):
#         batch = new_chunks[i:i + batch_size]
#         collection.add(
#             ids        = [c.chunk_id for c in batch],
#             documents  = [c.text for c in batch],
#             metadatas  = [{"source": c.source, "page": c.page,
#                            "parent_id": c.parent_id} for c in batch]
#         )
#         print(f"  ChromaDB: indexed {min(i+batch_size, len(new_chunks))}/{len(new_chunks)} child chunks")


# def build_bm25_index(chunks: List[Chunk]):
#     child_chunks = [c for c in chunks if c.chunk_type == "child"]

#     tokenized = [c.text.lower().split() for c in child_chunks]
#     bm25      = BM25Okapi(tokenized)

#     BM25_DIR.mkdir(parents=True, exist_ok=True)
#     index_path = BM25_DIR / "bm25.pkl"
#     meta_path  = BM25_DIR / "bm25_meta.json"

#     with open(index_path, "wb") as f:
#         pickle.dump(bm25, f)

#     meta = [{"chunk_id": c.chunk_id, "source": c.source,
#               "page": c.page, "parent_id": c.parent_id,
#               "text": c.text} for c in child_chunks]
#     with open(meta_path, "w", encoding="utf-8") as f:
#         json.dump(meta, f, ensure_ascii=False)

#     print(f"  BM25: indexed {len(child_chunks)} chunks → {index_path}")
#     return bm25


# def load_bm25_index():
#     index_path = BM25_DIR / "bm25.pkl"
#     meta_path  = BM25_DIR / "bm25_meta.json"

#     if not index_path.exists():
#         raise FileNotFoundError("BM25 index not found. Run indexer.py first.")

#     with open(index_path, "rb") as f:
#         bm25 = pickle.load(f)
#     with open(meta_path, "r", encoding="utf-8") as f:
#         meta = json.load(f)

#     return bm25, meta


# if __name__ == "__main__":
#     print("=" * 50)
#     print("  Week 1 — Building Dual Index")
#     print("=" * 50)

#     chunks_path = TESTS_DIR / "chunks.json"
#     if not chunks_path.exists():
#         print("chunks.json not found. Run ingestor.py first.")
#         exit(1)

#     print("\nLoading chunks from disk...")
#     chunks = load_chunks(chunks_path)
#     child_count = sum(1 for c in chunks if c.chunk_type == "child")
#     print(f"  Loaded {len(chunks)} total chunks ({child_count} child chunks to index)")

#     print("\nBuilding ChromaDB index (downloads embedding model on first run)...")
#     client     = get_chroma_client()
#     collection = get_or_create_collection(client)
#     index_to_chroma(collection, chunks)
#     print(f"  ChromaDB collection size: {collection.count()} chunks")

#     print("\nBuilding BM25 index...")
#     build_bm25_index(chunks)

#     print("\nTesting both indexes with a sample query...")
#     query = "what is the main topic of this document"

#     # Test ChromaDB
#     dense_results = collection.query(query_texts=[query], n_results=3)
#     print(f"\n  ChromaDB top 3 results for: '{query}'")
#     for i, (doc, meta) in enumerate(zip(
#         dense_results["documents"][0],
#         dense_results["metadatas"][0]
#     )):
#         print(f"    {i+1}. [{meta['source']} p{meta['page']}] {doc[:120]}...")

#     # Test BM25
#     bm25, meta_list = load_bm25_index()
#     tokenized_query = query.lower().split()
#     scores          = bm25.get_scores(tokenized_query)
#     top3_idx        = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

#     print(f"\n  BM25 top 3 results for: '{query}'")
#     for rank, idx in enumerate(top3_idx):
#         m = meta_list[idx]
#         print(f"    {rank+1}. [{m['source']} p{m['page']}] score={scores[idx]:.3f}  {m['text'][:120]}...")

#     print("\nIndexing complete. Both indexes ready.")
#     print("Next step: run src/retriever.py")

import json
import pickle
import sys
import numpy as np
from pathlib import Path
from typing import List

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))
from config import BM25_DIR, EMBED_MODEL, CHROMA_DIR
from src.ingestor import Chunk, load_chunks, TESTS_DIR

FAISS_DIR  = CHROMA_DIR          # reuse same path variable, different files
INDEX_FILE = FAISS_DIR / "faiss.index"
META_FILE  = FAISS_DIR / "faiss_meta.json"
EMBED_FILE = FAISS_DIR / "embedder.pkl"

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False,
                              batch_size=32, normalize_embeddings=True)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(chunks: List[Chunk]):
    child_chunks = [c for c in chunks if c.chunk_type == "child"]
    print(f"  Embedding {len(child_chunks)} child chunks...")

    texts      = [c.text for c in child_chunks]
    embeddings = embed_texts(texts)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product (cosine after normalization)
    index.add(embeddings)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    meta = [{"chunk_id": c.chunk_id, "source": c.source,
             "page": c.page, "parent_id": c.parent_id,
             "text": c.text} for c in child_chunks]
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"  FAISS: indexed {len(child_chunks)} chunks → {INDEX_FILE}")
    return index, meta


def load_faiss_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError("FAISS index not found. Run indexer.py first.")
    index = faiss.read_index(str(INDEX_FILE))
    
    # Handle missing metadata file gracefully (e.g., on fresh Streamlit Cloud deployment)
    if META_FILE.exists():
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = []
    
    return index, meta


def dense_search_faiss(index, meta, query: str, top_k: int = 20):
    embedder   = get_embedder()
    q_vec      = embedder.encode([query], normalize_embeddings=True)
    q_vec      = np.array(q_vec, dtype="float32")
    scores, ids = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        m = meta[idx]
        results.append({**m, "score": float(score), "method": "dense"})
    return results


def build_bm25_index(chunks: List[Chunk]):
    child_chunks = [c for c in chunks if c.chunk_type == "child"]
    tokenized    = [c.text.lower().split() for c in child_chunks]
    bm25         = BM25Okapi(tokenized)

    BM25_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    meta = [{"chunk_id": c.chunk_id, "source": c.source,
             "page": c.page, "parent_id": c.parent_id,
             "text": c.text} for c in child_chunks]
    with open(BM25_DIR / "bm25_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"  BM25: indexed {len(child_chunks)} chunks → {BM25_DIR / 'bm25.pkl'}")
    return bm25


def load_bm25_index():
    index_path = BM25_DIR / "bm25.pkl"
    meta_path  = BM25_DIR / "bm25_meta.json"
    if not index_path.exists():
        raise FileNotFoundError("BM25 index not found. Run indexer.py first.")
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    
    # Handle missing metadata file gracefully (e.g., on fresh Streamlit Cloud deployment)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = []
    
    return bm25, meta


def index_to_chroma(collection, chunks):
    build_faiss_index(chunks)


def get_chroma_client():
    return None


def get_or_create_collection(client):
    try:
        index, meta = load_faiss_index()
        return {"index": index, "meta": meta}
    except FileNotFoundError:
        return {"index": None, "meta": []}


if __name__ == "__main__":
    print("=" * 50)
    print("  Building Dual Index (FAISS + BM25)")
    print("=" * 50)

    chunks_path = TESTS_DIR / "chunks.json"
    if not chunks_path.exists():
        print("chunks.json not found. Run ingestor.py first.")
        exit(1)

    chunks = load_chunks(chunks_path)
    child  = sum(1 for c in chunks if c.chunk_type == "child")
    print(f"Loaded {len(chunks)} chunks ({child} child chunks)")

    print("\nBuilding FAISS index...")
    index, meta = build_faiss_index(chunks)
    print(f"  FAISS size: {index.ntotal} vectors")

    print("\nBuilding BM25 index...")
    build_bm25_index(chunks)

    print("\nTest query:")
    results = dense_search_faiss(index, meta, "what is machine learning", top_k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['source']} p{r['page']}] score={r['score']:.4f}  {r['text'][:100]}...")

    print("\nBoth indexes built.")
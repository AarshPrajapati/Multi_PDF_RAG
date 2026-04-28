import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).parent.parent))
from config import BM25_TOP_K, DENSE_TOP_K, RERANK_TOP_K, RERANK_MODEL
from src.indexer import get_chroma_client, get_or_create_collection, load_bm25_index


@dataclass
class RetrievedChunk:
    text:      str
    source:    str
    page:      int
    chunk_id:  str
    parent_id: str
    score:     float
    method:    str   # "dense", "bm25", or "hybrid"


def dense_search(collection, query: str, top_k: int = DENSE_TOP_K) -> List[RetrievedChunk]:
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks  = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append(RetrievedChunk(
            text      = doc,
            source    = meta["source"],
            page      = meta["page"],
            chunk_id  = results["ids"][0][len(chunks)],
            parent_id = meta["parent_id"],
            score     = 1 - dist,   # cosine distance → similarity
            method    = "dense"
        ))
    return chunks


def bm25_search(bm25, meta_list: List[Dict],
                query: str, top_k: int = BM25_TOP_K) -> List[RetrievedChunk]:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    chunks = []
    for idx in top_idx:
        if scores[idx] == 0:
            continue
        m = meta_list[idx]
        chunks.append(RetrievedChunk(
            text      = m["text"],
            source    = m["source"],
            page      = m["page"],
            chunk_id  = m["chunk_id"],
            parent_id = m["parent_id"],
            score     = float(scores[idx]),
            method    = "bm25"
        ))
    return chunks


def reciprocal_rank_fusion(dense_results: List[RetrievedChunk],
                           bm25_results:  List[RetrievedChunk],
                           k: int = 60) -> List[RetrievedChunk]:
    # k=60 is the standard RRF constant from the original paper
    # Score = sum of 1/(k + rank) across all lists a chunk appears in
    rrf_scores: Dict[str, float] = {}
    chunk_map:  Dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id]  = chunk

    for rank, chunk in enumerate(bm25_results):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id]  = chunk

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for cid in sorted_ids:
        c = chunk_map[cid]
        merged.append(RetrievedChunk(
            text      = c.text,
            source    = c.source,
            page      = c.page,
            chunk_id  = c.chunk_id,
            parent_id = c.parent_id,
            score     = rrf_scores[cid],
            method    = "hybrid"
        ))
    return merged


def rerank(query: str, chunks: List[RetrievedChunk],
           top_k: int = RERANK_TOP_K,
           model_name: str = RERANK_MODEL) -> List[RetrievedChunk]:
    if not chunks:
        return []

    reranker = CrossEncoder(model_name)
    pairs    = [[query, c.text] for c in chunks]
    scores   = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score  = float(score)
        chunk.method = "reranked"

    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return reranked[:top_k]


def get_parent_texts(top_chunks: List[RetrievedChunk],
                     all_meta: List[Dict]) -> List[RetrievedChunk]:
    parent_map = {m["chunk_id"]: m for m in all_meta if m.get("chunk_type") == "parent"}

    enriched = []
    for chunk in top_chunks:
        parent = parent_map.get(chunk.parent_id)
        if parent:
            chunk.text = parent["text"]   # swap child text → parent text
        enriched.append(chunk)
    return enriched


def retrieve(query: str, collection, bm25, meta_list: List[Dict],
             use_rerank: bool = True) -> List[RetrievedChunk]:
    dense_results = dense_search(collection, query)
    bm25_results  = bm25_search(bm25, meta_list, query)
    merged        = reciprocal_rank_fusion(dense_results, bm25_results)

    if use_rerank:
        final = rerank(query, merged)
    else:
        final = merged[:RERANK_TOP_K]

    return final


if __name__ == "__main__":
    import json
    from config import TESTS_DIR

    print("=" * 50)
    print("  Week 2 — Hybrid Retrieval Test")
    print("=" * 50)

    client     = get_chroma_client()
    collection = get_or_create_collection(client)
    bm25, meta = load_bm25_index()

    # Load all chunks for parent lookup
    chunks_path = TESTS_DIR / "chunks.json"
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks_raw = json.load(f)

    test_queries = [
        "what is the main topic of this document",
        "summarize the key findings",
        "what methods were used",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 45)

        # Without reranking
        merged = reciprocal_rank_fusion(
            dense_search(collection, query),
            bm25_search(bm25, meta, query)
        )

        # With reranking
        final = rerank(query, merged[:20])

        print(f"  After hybrid search : {len(merged)} candidates")
        print(f"  After re-ranking    : {len(final)} final chunks")
        print()
        for i, chunk in enumerate(final):
            print(f"  {i+1}. [{chunk.source} p{chunk.page}] score={chunk.score:.4f}")
            print(f"     {chunk.text[:150]}...")
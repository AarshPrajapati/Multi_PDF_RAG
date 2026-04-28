import json
import sys
from pathlib import Path
from typing import List
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))
from config import OPENAI_API_KEY, LLM_MODEL, TESTS_DIR
from src.retriever import RetrievedChunk, retrieve, get_chroma_client, get_or_create_collection, load_bm25_index


client = OpenAI(api_key=OPENAI_API_KEY)


ANSWER_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.

RULES:
- Only use information from the context below
- If the context does not contain the answer, say "I cannot find this in the provided documents"
- Always cite the source PDF and page number for every claim
- Format citations as [filename, p.X]

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def format_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Source {i+1}: {chunk.source}, page {chunk.page}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> dict:
    context = format_context(chunks)
    prompt  = ANSWER_PROMPT.format(context=context, question=question)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )

    answer = response.choices[0].message.content.strip()

    sources = []
    seen    = set()
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            sources.append({"source": chunk.source, "page": chunk.page})
            seen.add(key)

    return {
        "question":    question,
        "answer":      answer,
        "sources":     sources,
        "model":       LLM_MODEL,
        "chunks_used": len(chunks)
    }


def ask(question: str, collection, bm25, meta_list: List[dict]) -> dict:
    chunks = retrieve(question, collection, bm25, meta_list)
    result = generate_answer(question, chunks)
    return result


def print_result(result: dict):
    print(f"\nQuestion : {result['question']}")
    print(f"{'─'*55}")
    print(f"Answer   :\n{result['answer']}")
    print(f"\nSources  :")
    for s in result["sources"]:
        print(f"  - {s['source']}  page {s['page']}")
    print(f"\nChunks used : {result['chunks_used']}  |  Model : {result['model']}")


if __name__ == "__main__":
    print("="*50)
    print("  Week 3 — Generation with Citations")
    print("="*50)

    client_db  = get_chroma_client()
    collection = get_or_create_collection(client_db)
    bm25, meta = load_bm25_index()

    questions = [
        "What is artificial intelligence?",
        "What is the difference between machine learning and deep learning?",
        "Why is studying AI important?",
    ]

    results = []
    for q in questions:
        print(f"\nAsking: {q}")
        result = ask(q, collection, bm25, meta)
        print_result(result)
        results.append(result)

    out = TESTS_DIR / "generation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out}")
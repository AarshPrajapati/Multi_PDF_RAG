import json
import sys
from pathlib import Path
from typing import List
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))
from config import OPENAI_API_KEY, LLM_MODEL, TESTS_DIR
from src.retriever import retrieve, get_chroma_client, get_or_create_collection, load_bm25_index


client = OpenAI(api_key=OPENAI_API_KEY)


MCQ_PROMPT = """You are an expert educator creating exam questions strictly from the provided study material.

RULES:
- Every question MUST be answerable from the context below
- Do NOT use any outside knowledge
- Each question must cite the page number it came from
- Return ONLY valid JSON, no extra text

CONTEXT:
{context}

TOPIC: {topic}
DIFFICULTY: {difficulty}
NUMBER OF QUESTIONS: {num_questions}

Generate {num_questions} multiple choice questions. Return this exact JSON format:
{{
  "questions": [
    {{
      "question": "question text here",
      "options": {{
        "A": "option A",
        "B": "option B",
        "C": "option C",
        "D": "option D"
      }},
      "correct_answer": "A",
      "explanation": "why this answer is correct",
      "source_page": 12
    }}
  ]
}}"""


SHORT_ANSWER_PROMPT = """You are an expert educator creating exam questions strictly from the provided study material.

RULES:
- Every question MUST be answerable from the context below
- Do NOT use any outside knowledge
- Each question must cite the page number it came from
- Return ONLY valid JSON, no extra text

CONTEXT:
{context}

TOPIC: {topic}
DIFFICULTY: {difficulty}
NUMBER OF QUESTIONS: {num_questions}

Generate {num_questions} short answer questions. Return this exact JSON format:
{{
  "questions": [
    {{
      "question": "question text here",
      "model_answer": "expected answer in 2-3 sentences",
      "key_points": ["point 1", "point 2", "point 3"],
      "source_page": 12
    }}
  ]
}}"""


ESSAY_PROMPT = """You are an expert educator creating exam questions strictly from the provided study material.

RULES:
- Every question MUST be answerable from the context below
- Do NOT use any outside knowledge
- Return ONLY valid JSON, no extra text

CONTEXT:
{context}

TOPIC: {topic}
DIFFICULTY: {difficulty}

Generate 1 essay question. Return this exact JSON format:
{{
  "questions": [
    {{
      "question": "essay question here",
      "guidance": "what a good answer should cover",
      "key_themes": ["theme 1", "theme 2", "theme 3"],
      "suggested_length": "300-500 words",
      "source_pages": [12, 15, 18]
    }}
  ]
}}"""


def get_prompt(q_type: str) -> str:
    return {"mcq": MCQ_PROMPT, "short": SHORT_ANSWER_PROMPT, "essay": ESSAY_PROMPT}[q_type]


def generate_assignment(topic: str, difficulty: str, q_type: str,
                        num_questions: int, collection, bm25, meta_list) -> dict:

    query  = f"{topic} {difficulty}"
    chunks = retrieve(query, collection, bm25, meta_list)

    context = "\n\n---\n\n".join(
        f"[Page {c.page}]\n{c.text}" for c in chunks
    )

    prompt = get_prompt(q_type).format(
        context       = context,
        topic         = topic,
        difficulty    = difficulty,
        num_questions = num_questions
    )

    response = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.4,
        max_tokens  = 2000
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)

    sources = list({(c.source, c.page) for c in chunks})
    sources.sort(key=lambda x: x[1])

    return {
        "topic":      topic,
        "difficulty": difficulty,
        "type":       q_type,
        "questions":  parsed["questions"],
        "sources":    [{"source": s, "page": p} for s, p in sources],
        "model":      LLM_MODEL
    }


def print_assignment(result: dict):
    q_type = result["type"]
    print(f"\n{'='*55}")
    print(f"Topic      : {result['topic']}")
    print(f"Difficulty : {result['difficulty']}")
    print(f"Type       : {q_type.upper()}")
    print(f"{'='*55}")

    for i, q in enumerate(result["questions"], 1):
        print(f"\nQ{i}. {q['question']}")

        if q_type == "mcq":
            for letter, text in q["options"].items():
                marker = "✓" if letter == q["correct_answer"] else " "
                print(f"   {marker} {letter}. {text}")
            print(f"\n   Answer      : {q['correct_answer']}")
            print(f"   Explanation : {q['explanation']}")
            print(f"   Source      : page {q['source_page']}")

        elif q_type == "short":
            print(f"   Model answer : {q['model_answer']}")
            print(f"   Key points   : {', '.join(q['key_points'])}")
            print(f"   Source       : page {q['source_page']}")

        elif q_type == "essay":
            print(f"   Guidance     : {q['guidance']}")
            print(f"   Key themes   : {', '.join(q['key_themes'])}")
            print(f"   Length       : {q['suggested_length']}")
            print(f"   Sources      : pages {q['source_pages']}")

    print(f"\nSources used:")
    for s in result["sources"]:
        print(f"  - {s['source']}  page {s['page']}")


if __name__ == "__main__":
    print("="*50)
    print("  Week 5 — Assignment Generator")
    print("="*50)

    client_db  = get_chroma_client()
    collection = get_or_create_collection(client_db)
    bm25, meta = load_bm25_index()

    print("\n[1/3] Generating MCQ questions...")
    mcq = generate_assignment(
        topic         = "machine learning types",
        difficulty    = "intermediate",
        q_type        = "mcq",
        num_questions = 3,
        collection    = collection,
        bm25          = bm25,
        meta_list     = meta
    )
    print_assignment(mcq)

    print("\n[2/3] Generating short answer questions...")
    short = generate_assignment(
        topic         = "artificial intelligence definition",
        difficulty    = "beginner",
        q_type        = "short",
        num_questions = 2,
        collection    = collection,
        bm25          = bm25,
        meta_list     = meta
    )
    print_assignment(short)

    print("\n[3/3] Generating essay question...")
    essay = generate_assignment(
        topic         = "future of AI and ethics",
        difficulty    = "advanced",
        q_type        = "essay",
        num_questions = 1,
        collection    = collection,
        bm25          = bm25,
        meta_list     = meta
    )
    print_assignment(essay)

    out = TESTS_DIR / "assignments.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump([mcq, short, essay], f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out}")
import fitz
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, PARENT_SIZE, TESTS_DIR


@dataclass
class Chunk:
    text: str
    source: str        # PDF filename
    page: int          # page number (1-indexed)
    chunk_id: str      # unique id: filename_page_chunknum
    parent_id: str     # parent chunk id (for parent-doc retrieval)
    chunk_type: str    # "child" or "parent"


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 newlines in a row
    text = re.sub(r'[ \t]+', ' ', text)       # collapse spaces/tabs
    text = re.sub(r'\x00', '', text)           # remove null bytes
    return text.strip()


def extract_pages(pdf_path: Path) -> List[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = clean_text(text)
        if len(text) < 50:       # skip nearly-empty pages
            continue
        pages.append({
            "page": page_num + 1,
            "text": text,
            "source": pdf_path.name
        })
    doc.close()
    return pages


def split_into_chunks(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 30:   # skip tiny fragments
            chunks.append(chunk)
        start += size - overlap
    return chunks


def build_chunks(pdf_path: Path) -> List[Chunk]:
    pages = extract_pages(pdf_path)
    stem = pdf_path.stem

    all_chunks = []
    parent_idx = 0

    for page_data in pages:
        page_num = page_data["page"]
        text     = page_data["text"]
        source   = page_data["source"]

        # Build parent chunks from this page
        parent_texts = split_into_chunks(text, PARENT_SIZE, CHUNK_OVERLAP)

        for p_text in parent_texts:
            parent_id = f"{stem}_p{page_num}_par{parent_idx}"

            # Store parent chunk
            parent_chunk = Chunk(
                text       = p_text,
                source     = source,
                page       = page_num,
                chunk_id   = parent_id,
                parent_id  = parent_id,
                chunk_type = "parent"
            )
            all_chunks.append(parent_chunk)

            # Build child chunks from parent
            child_texts = split_into_chunks(p_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for c_idx, c_text in enumerate(child_texts):
                child_id = f"{parent_id}_ch{c_idx}"
                child_chunk = Chunk(
                    text       = c_text,
                    source     = source,
                    page       = page_num,
                    chunk_id   = child_id,
                    parent_id  = parent_id,
                    chunk_type = "child"
                )
                all_chunks.append(child_chunk)

            parent_idx += 1

    return all_chunks


def ingest_all_pdfs(pdf_dir: Path = DATA_DIR) -> List[Chunk]:
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        print("Put your PDF files in the data/pdfs/ folder and run again.")
        return []

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        chunks = build_chunks(pdf_path)
        child_count  = sum(1 for c in chunks if c.chunk_type == "child")
        parent_count = sum(1 for c in chunks if c.chunk_type == "parent")
        print(f"    Pages extracted  : {len(set(c.page for c in chunks))}")
        print(f"    Parent chunks    : {parent_count}")
        print(f"    Child chunks     : {child_count}")
        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(chunks: List[Chunk], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(c) for c in chunks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(chunks)} chunks → {out_path}")


def load_chunks(path: Path) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**d) for d in data]


if __name__ == "__main__":
    print("=" * 50)
    print("  Week 1 — PDF Ingestion")
    print("=" * 50)

    chunks = ingest_all_pdfs()

    if not chunks:
        exit(1)

    child_chunks  = [c for c in chunks if c.chunk_type == "child"]
    parent_chunks = [c for c in chunks if c.chunk_type == "parent"]

    print(f"\nTotal PDFs processed : {len(set(c.source for c in chunks))}")
    print(f"Total parent chunks  : {len(parent_chunks)}")
    print(f"Total child chunks   : {len(child_chunks)}")

    # Save to tests/ folder for inspection
    out_path = TESTS_DIR / "chunks.json"
    TESTS_DIR.mkdir(exist_ok=True)
    save_chunks(chunks, out_path)

    # Print 2 sample child chunks so you can verify quality
    print("\n── Sample child chunk 1 ──")
    print(f"Source : {child_chunks[0].source}  Page {child_chunks[0].page}")
    print(f"ID     : {child_chunks[0].chunk_id}")
    print(f"Text   : {child_chunks[0].text[:300]}...")

    if len(child_chunks) > 1:
        print("\n── Sample child chunk 2 ──")
        print(f"Source : {child_chunks[1].source}  Page {child_chunks[1].page}")
        print(f"Text   : {child_chunks[1].text[:300]}...")
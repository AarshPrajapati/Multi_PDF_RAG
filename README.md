# Multi-PDF RAG — Advanced Retrieval-Augmented Generation

Upload any PDFs → ask questions with page citations → generate assignments and quizzes grounded in your documents.

## Features
- Hybrid search (BM25 + semantic) with Reciprocal Rank Fusion
- Cross-encoder re-ranking for precision
- Parent-document retrieval for full context
- GPT-4o-mini generation with page citations
- MCQ, short answer, and essay assignment generator

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenAI key
4. Add PDFs to `data/pdfs/`
5. Run: `streamlit run app.py`
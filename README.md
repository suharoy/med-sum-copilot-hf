# Medical Research Summarizer 

A *local, offline GPT-like assistant* that summarizes and answers questions from uploaded *medical research papers* — built entirely with open-source Hugging Face models, no paid APIs or GPU required.

---

## ⚙ Overview

This project demonstrates how to build a lightweight *Retrieval-Augmented Generation (RAG)* pipeline on CPU using open-source models.

Users can:
- Upload research PDFs  
- Ask domain-specific questions (e.g., “What treatment is recommended?”)  
- Receive concise, citation-linked answers generated locally  

All processing — PDF parsing, retrieval, embedding, and summarization — happens fully on your machine.

---

## 🧩 Architecture

apps/ ├── api/ │   ├── core/ │   │   └── config.py        ← Loads global .env and paths │   ├── nlp/ │   │   └── parse_pdf.py     ← Converts PDFs → JSON (sections & sentences) │   └── rag/ │       ├── embed.py         ← Sentence embeddings via MiniLM │       ├── retrieve.py      ← Hybrid BM25 + semantic retriever │       └── generate.py      ← Summarization / answer generation └── web/ └── app.py               ← Streamlit web interface

.env / global.env            ← API keys or model names (never commit) requirements.txt             ← Dependencies README.md                    ← This file

---

## 🧠 How It Works

### 1️⃣ PDF Parsing (parse_pdf.py)
- Extracts text with *PyPDF2* and basic regex cleaning.
- Splits content into structured sections:

```json
{
  "paper_id": "breast-cancer-meta-analysis",
  "sections": [
    {"name": "Abstract", "text": "..."},
    {"name": "Results",  "text": "..."},
    {"name": "Conclusion","text": "..."}
  ]
}

Parsed outputs live in data/parsed/.


---

2️⃣ Embedding & Retrieval (embed.py, retrieve.py)

Uses Sentence-Transformers MiniLM-L6-v2 for semantic vectors.

BM25 ranks lexical overlap; MMR ensures diverse top-k evidence.

Combines lexical + semantic + section weighting for better context selection.



---

3️⃣ Summarization (generate.py)

Abstractive summarization via DistilBART-CNN (default).

Chunked batching for long texts to stay within CPU limits.

Cleans citations and formats readable paragraphs.

Falls back to extractive summaries if abstractive fails.



---

4️⃣ Frontend (app.py)

A Streamlit dashboard providing:

Paper selector

Audience switch (Expert / Patient)

Question box

Instant answer + cited evidence view


Everything executes locally — no network calls.


---

🏗 Core Concepts

Concept	Explanation

RAG (Retrieval-Augmented Generation)	Combines search + generation for grounded answers.
BM25	Classical keyword ranking (term-frequency / inverse-doc-freq).
Sentence Transformer	Embeds sentences into dense semantic vectors.
MMR	Maximal Marginal Relevance to keep retrieved evidence diverse.
DistilBART	Lightweight summarization transformer ideal for CPUs.



---

💡 Design Philosophy

Educational → transparent end-to-end RAG example

Local-first → no API keys or GPU dependency

Extensible → swap models or add new prompts easily

Transparent → inspect every intermediate step



---

📁 Workflow Example

1️⃣ Parse your PDF

python -m apps.api.nlp.parse_pdf "data/raw/Immunotherapy_Cancer_Treatment.pdf"

Produces data/parsed/immunotherapy_cancer_treatment.json.

2️⃣ Run the app

streamlit run apps/web/app.py

3️⃣ Use the web UI

Pick a paper ID

Choose audience

Ask questions like “What are the main findings?”

View summarized answers + source snippets



---

⚡ Performance Notes

Metric	Approx Value

CPU latency	5 – 15 s per query
RAM usage	< 1 GB
Disk cache	~500 MB (HF models + embeddings)


Tips to speed up

Reduce top_k in retriever (e.g. 6 → 4)

Lower max_words chunk size in generate.py

Switch to sshleifer/distilbart-cnn-12-6 (smaller model)

Run on GPU if available → set device=0



---

🧰 Extending the App

Goal	File to edit	Hint

Use another summarizer	generate.py	Change model in pipeline()
Tune section weights	retrieve.py	Adjust SECTION_WEIGHTS
Add new prompt style	generate.py	Edit summarize_paper()
Enable GPU	generate.py	Set device=0 in pipeline init
Cache retrieval results	retrieve.py	Decorate search() with lru_cache



---

🧠 Learning Outcomes

By studying this repo you’ll grasp:

How RAG pipelines combine retrieval + generation

Integrating Hugging Face pipelines for CPU inference

Designing transparent, educational AI tools

Structuring research summarization apps end-to-end



---

🙌 Authors & Credits

Developed by Sriparno Ganguly (K2E7)
Educational project showcasing open-source RAG for medical literature.

Models from Hugging Face Hub
Example papers from Indian Journal of Medical Research (IJMR).


---

🚀 Quick Start

# clone
git clone https://github.com/<your-username>/medical-research-summarizer
cd medical-research-summarizer

# create venv
python -m venv .venv
.\.venv\Scripts\activate      # on Windows
# or
source .venv/bin/activate     # on Linux/Mac

# install dependencies
pip install -r requirements.txt

# parse your PDFs
python -m apps.api.nlp.parse_pdf "data/raw/<your_paper>.pdf"

# run the local web app
streamlit run apps/web/app.py

Once launched, open http://localhost:8501 in your browser.

---

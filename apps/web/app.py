import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import streamlit as st, json
from apps.api.nlp.parse_pdf import parse_pdf_to_sections
from apps.api.rag.retrieve import SimpleStore
from apps.api.rag.generate import answer_from_context
from apps.api.core.config import settings

st.set_page_config(page_title="Medical Research Summarizer (HF)", page_icon="🧪", layout="wide")
st.title("🧪 Medical Research Summarizer — Hugging Face (CPU / Free)")
st.caption("Upload PDFs → Parse → Ask questions → Summaries")

with st.sidebar:
    st.header("Settings")
    st.write("Embed model:", settings.HF_EMBED_MODEL)
    st.write("Summarizer model:", settings.HF_SUMMARIZER_MODEL)
    st.write("Storage:", settings.STORAGE_DIR)

# Upload
st.subheader("1) Upload PDFs")
files = st.file_uploader("Drag & drop PDFs", type=["pdf"], accept_multiple_files=True)
if files:
    out_dir = Path(settings.STORAGE_DIR) / "pdfs"; out_dir.mkdir(parents=True, exist_ok=True)
    for f in files: (out_dir / f.name).write_bytes(f.read())
    st.success(f"Uploaded {len(files)} file(s).")

# Ingest
st.subheader("2) Parse / Refresh")
if st.button("Parse now"):
    pdf_dir = Path(settings.STORAGE_DIR) / "pdfs"
    out_dir = Path(settings.STORAGE_DIR) / "parsed"; out_dir.mkdir(parents=True, exist_ok=True)
    n=0
    for pdf in pdf_dir.glob("*.pdf"):
        parsed = parse_pdf_to_sections(pdf)
        (out_dir / f"{pdf.stem}.json").write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8"); n+=1
    st.success(f"Ingested {n} file(s).")

parsed_dir = Path(settings.STORAGE_DIR) / "parsed"
papers = sorted(parsed_dir.glob("*.json"))
with st.expander("📚 Library", expanded=True):
    if not papers: st.info("No parsed papers yet.")
    else:
        for p in papers:
            j=json.loads(p.read_text(encoding="utf-8"))
            st.markdown(f"- **{j.get('title', p.stem)}** — `paper_id`: `{j.get('paper_id', p.stem)}` — sections: {len(j.get('sections', []))}")

st.divider()

# Ask
st.subheader("3) Ask a question")
q = st.text_input("Your question", placeholder="What is the main conclusion?")
k = st.slider("Max evidence chunks", 2, 10, 6)
if st.button("Get answer", type="primary") and q:
    with st.status("Working...", expanded=True):
        store = SimpleStore(parsed_dir); ctxs = store.search(q, top_k=k)
        if not ctxs: st.error("No relevant evidence found.")
        else:
            ans = answer_from_context(q, ctxs)
            st.markdown("### ✅ Answer"); st.write(ans)
            with st.expander("📎 Evidence & Citations"):
                for i,c in enumerate(ctxs,1):
                    st.markdown(f"**[{i}]** `{c['meta']['source']}` — *{c['meta']['section']}*")
                    st.caption(c['text'][:600]+('...' if len(c['text'])>600 else ''))

st.divider()

# Summarize
st.subheader("4) Summarize a paper")
ids=[p.stem for p in papers]
if ids:
    chosen = st.selectbox("Choose paper_id", ids, index=0)
    mode = st.radio("Audience", ["expert","patient"], horizontal=True)
    if st.button("Summarize"):
        j=json.loads((parsed_dir/f"{chosen}.json").read_text(encoding="utf-8"))
        ctxs=[{"text":s["text"],"meta":{"source":f"{chosen}.json","section":s["name"]}} for s in j.get("sections",[])]
        prompt=f"Summarize this paper for a {'researcher' if mode=='expert' else 'patient'}."
        summ = answer_from_context(prompt, ctxs)
        st.markdown("### 🧾 Summary"); st.write(summ)
else:
    st.info("No papers to summarize yet.")

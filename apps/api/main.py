from fastapi import FastAPI
from .routes import upload, ingest, ask, summarize
app = FastAPI(title="Medical Research Summarizer (HF)", version="1.0.0")
app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])
app.include_router(summarize.router, prefix="/summarize", tags=["summarize"])
@app.get("/health")
def health():
    return {"status":"ok"}

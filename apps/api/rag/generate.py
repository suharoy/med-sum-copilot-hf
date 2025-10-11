import time
import re
from typing import List
from transformers import pipeline

_SUMMARIZER = None

def get_summarizer():
    """
    Returns a cached summarization pipeline using a light Hugging Face model.
    Loads only once and stays in memory.
    """
    global _SUMMARIZER
    if _SUMMARIZER is None:
        print("🔹 Loading summarizer model (DistilBART)...")
        _SUMMARIZER = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",  # faster, smaller
            device=-1  # CPU only
        )
    return _SUMMARIZER


def chunk_text(text: str, max_words: int = 600) -> List[str]:
    """
    Split long text into smaller chunks (for models with token limits).
    """
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def summarize_text(text: str) -> str:
    """
    Performs chunked summarization and concatenates results.
    """
    summarizer = get_summarizer()
    chunks = chunk_text(text)
    summaries = []
    for i, chunk in enumerate(chunks):
        start = time.time()
        try:
            result = summarizer(
                chunk,
                max_length=120,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(result)
            print(f"✅ Chunk {i+1}/{len(chunks)} summarized in {time.time()-start:.2f}s")
        except Exception as e:
            print(f"⚠ Chunk {i+1} failed: {e}")
    return " ".join(summaries).strip()


def clean_summary(text: str) -> str:
    """
    Minor cleanup for summaries (removes references, weird spacing, etc.)
    """
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def summarize_paper(context: str, question: str = None) -> str:
    """
    Generates a concise answer or summary based on given context and question.
    """
    start_time = time.time()
    summarizer = get_summarizer()

    if not context or len(context.strip()) == 0:
        return "No context available for summarization."

    # If question provided, make the summary focused
    if question:
        prompt = f"Summarize the following medical research text focusing on the question: '{question}'."
    else:
        prompt = "Summarize the following medical research text concisely and factually."

    text_to_summarize = f"{prompt}\n\n{context}"

    try:
        summary = summarize_text(text_to_summarize)
        summary = clean_summary(summary)
        print(f"Total summarization time: {time.time() - start_time:.2f}s")
        return summary
    except Exception as e:
        print(f"Summarization failed, falling back to extractive summary: {e}")
        return extractive_summary_fallback(context)


def extractive_summary_fallback(text: str, max_sentences: int = 4) -> str:
    """
    Fallback summarizer: returns top few sentences as bullet points.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    top_sentences = sentences[:max_sentences]
    return "\n".join([f"• {s.strip()}" for s in top_sentences if len(s.strip()) > 20])


# Example usage:
if __name__ == "__main__":
    example_text = """
    Breast cancer remains a leading cause of death worldwide. Recent studies 
    indicate the potential of immunotherapy and targeted treatments. This study 
    evaluates meta-analytical results across several cohorts to establish the 
    prognostic impact of the kinesin superfamily proteins in cancer progression.
    """
    print(summarize_paper(example_text, question="What are the new treatments for breast cancer?"))
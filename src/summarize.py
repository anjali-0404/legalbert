from __future__ import annotations

import warnings

from transformers import pipeline

def _fallback_summary(text: str, max_chars: int = 280) -> str:
    s = " ".join(text.strip().split())
    return (s[: max_chars - 1] + "…") if len(s) > max_chars else s


try:
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
    )
except Exception as exc:
    warnings.warn(
        "Could not load summarization model (likely offline). "
        "Falling back to a simple extractive summary.\n"
        f"Details: {exc}"
    )
    summarizer = None

text = """
We collect personal information and share it with advertising partners.
Cookies are used to track user activity across websites.
"""

if summarizer is None:
    print(_fallback_summary(text))
else:
    summary = summarizer(text, max_length=80, min_length=30)
    print(summary[0]["summary_text"])

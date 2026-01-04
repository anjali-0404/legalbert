from __future__ import annotations

from pathlib import Path
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = [
    "DATA_COLLECTION","DATA_SHARING","THIRD_PARTY",
    "COOKIES","TRACKING","RETENTION","USER_RIGHTS","CONSENT_FORCED"
]

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = ROOT / "models" / "legalbert"
FALLBACK_MODEL_ID = "nlpaueb/legal-bert-base-uncased"


def _heuristic_label(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["third party", "third-party", "advertiser", "partner", "affiliat"]):
        return "THIRD_PARTY"
    if any(k in t for k in ["share", "sell", "disclose", "transfer"]):
        return "DATA_SHARING"
    if any(k in t for k in ["cookie", "cookies"]):
        return "COOKIES"
    if any(k in t for k in ["track", "tracking", "analytics", "pixel", "beacon"]):
        return "TRACKING"
    if any(k in t for k in ["retain", "retention", "keep", "store for"]):
        return "RETENTION"
    if any(k in t for k in ["delete", "access", "opt out", "opt-out", "rights"]):
        return "USER_RIGHTS"
    if any(k in t for k in ["consent", "by using", "must agree", "required"]):
        return "CONSENT_FORCED"
    if any(k in t for k in ["collect", "gather", "personal information", "data we collect"]):
        return "DATA_COLLECTION"
    return "DATA_COLLECTION"


def _load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if LOCAL_MODEL_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(LOCAL_MODEL_DIR), local_files_only=True
        )
        return tokenizer, model.to(device).eval(), device

    try:
        warnings.warn(
            f"Local model not found at {LOCAL_MODEL_DIR}. Using fallback model '{FALLBACK_MODEL_ID}'."
        )
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_MODEL_ID, num_labels=len(LABELS)
        )
        return tokenizer, model.to(device).eval(), device
    except Exception as exc:
        warnings.warn(
            "Could not load a Transformers model (likely offline/no access). "
            "Falling back to a simple heuristic classifier.\n"
            f"Details: {exc}"
        )
        return None, None, device

def analyze(text: str) -> str:
    tokenizer, model, device = _load_model_and_tokenizer()
    if tokenizer is None or model is None:
        return _heuristic_label(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return LABELS[int(outputs.logits.argmax().item())]


if __name__ == "__main__":
    print(analyze("We may share your data with third parties."))

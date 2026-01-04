from __future__ import annotations

from pathlib import Path
import warnings

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

LABELS = [
    "DATA_COLLECTION",
    "DATA_SHARING",
    "THIRD_PARTY",
    "COOKIES",
    "TRACKING",
    "RETENTION",
    "USER_RIGHTS",
    "CONSENT_FORCED",
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

try:
    if LOCAL_MODEL_DIR.exists():
        classifier = pipeline(
            "text-classification",
            model=str(LOCAL_MODEL_DIR),
            tokenizer=str(LOCAL_MODEL_DIR),
        )
    else:
        warnings.warn(
            f"Local model not found at {LOCAL_MODEL_DIR}. Using fallback model '{FALLBACK_MODEL_ID}'."
        )
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_MODEL_ID, num_labels=len(LABELS)
        )
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
except Exception as exc:
    classifier = None
    warnings.warn(
        "Could not initialize Transformers classifier; falling back to heuristic labels.\n"
        f"Details: {exc}"
    )

tests = [
    "We share your data with advertisers.",
    "Users can delete their personal data."
]

for t in tests:
    print(t)
    if classifier is None:
        print({"label": _heuristic_label(t)})
    else:
        out = classifier(t)
        try:
            if isinstance(out, list) and out and isinstance(out[0], dict):
                label = out[0].get("label")
                if isinstance(label, str) and label.startswith("LABEL_"):
                    idx = int(label.split("_", 1)[1])
                    if 0 <= idx < len(LABELS):
                        out[0]["label"] = LABELS[idx]
        except Exception:
            pass
        print(out)

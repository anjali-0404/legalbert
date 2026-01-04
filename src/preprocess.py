from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[1]

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    warnings.warn(
        "spaCy model 'en_core_web_sm' is not installed. "
        "Falling back to a blank English pipeline with a sentencizer. "
        "(Sentence splitting may be less accurate.)"
    )
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

LABEL_MAP = {
    "DATA_COLLECTION": 0,
    "DATA_SHARING": 1,
    "THIRD_PARTY": 2,
    "COOKIES": 3,
    "TRACKING": 4,
    "RETENTION": 5,
    "USER_RIGHTS": 6,
    "CONSENT_FORCED": 7
}

def split_clauses(text):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 25]

raw_path = ROOT / "data" / "raw" / "opp115.json"
processed_dir = ROOT / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

with raw_path.open("r", encoding="utf-8") as f:
    raw = json.load(f)

samples = []
for item in raw:
    clauses = split_clauses(item["text"])
    for clause in clauses:
        samples.append({
            "text": clause,
            "label": LABEL_MAP[item["label"]]
        })

random.shuffle(samples)

n = len(samples)
train = samples[:int(0.7*n)]
val   = samples[int(0.7*n):int(0.85*n)]
test  = samples[int(0.85*n):]

json.dump(train, (processed_dir / "train.json").open("w", encoding="utf-8"), indent=2)
json.dump(val, (processed_dir / "val.json").open("w", encoding="utf-8"), indent=2)
json.dump(test, (processed_dir / "test.json").open("w", encoding="utf-8"), indent=2)

print("✅ Preprocessing completed")

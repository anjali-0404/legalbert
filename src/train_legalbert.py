from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL = "nlpaueb/legal-bert-base-uncased"

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train.json"
VAL_PATH = ROOT / "data" / "processed" / "val.json"
OUT_DIR = ROOT / "models" / "legalbert"

if not TRAIN_PATH.exists() or not VAL_PATH.exists():
    raise SystemExit(
        "Processed data not found. Run preprocess first:\n"
        "  python src/preprocess.py\n"
        f"Missing: {TRAIN_PATH if not TRAIN_PATH.exists() else ''} {VAL_PATH if not VAL_PATH.exists() else ''}"
    )

dataset = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_PATH),
        "validation": str(VAL_PATH)
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids","attention_mask","label"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=8)

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=bool(torch.cuda.is_available())
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))

print("✅ Legal-BERT training completed")

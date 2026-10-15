"""Phase 1: data loading.

Filters dair-ai/emotion to the paper's 4 emotions (anger, fear, joy, sadness),
saves processed CSVs, and provides a Dataset + collate function.

Run from the project root to prepare the data:  python -m src.data
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import Dataset

import config

SPLIT_FILES = {
    "train": "emotion_train.csv",
    "val": "emotion_validation.csv",
    "test": "emotion_test.csv",
}


def prepare_processed() -> dict:
    """Read raw CSVs, keep 4 emotions, remap labels to 0..3, save to data/processed/."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, fname in SPLIT_FILES.items():
        df = pd.read_csv(config.DATA_RAW / fname)
        df["emotion"] = df["label"].map(config.DAIR_ID2NAME)
        df = df[df["emotion"].isin(config.EMOTIONS)].copy()
        df["label"] = df["emotion"].map(config.LABEL2ID)
        df = df[["text", "label", "emotion"]].reset_index(drop=True)
        df.to_csv(config.DATA_PROCESSED / f"{split}.csv", index=False)
        out[split] = df
    return out


def load_split(split: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED / f"{split}.csv"
    if not path.exists():
        prepare_processed()
    return pd.read_csv(path)


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = [int(x) for x in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.labels[i],
        }


def make_collate(pad_token_id: int):
    """Pads each batch to the longest sequence in that batch."""
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, mask = [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_token_id] * pad)
            mask.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor([b["labels"] for b in batch]),
        }
    return collate


if __name__ == "__main__":
    splits = prepare_processed()
    for name, df in splits.items():
        print(f"\n{name}: {len(df)} rows")
        print(df["emotion"].value_counts().to_string())
    print("\nSample rows:")
    print(splits["train"].head(5).to_string())
    print("\nSaved to data/processed/. Next: python -m src.train")

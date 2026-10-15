"""Phase 1: fine-tune an emotion classifier (baseline). Reused later for CDA.

Examples (run from the project root):
  python -m src.train                                   # baseline, seed 0
  python -m src.train --seed 1 --run_name baseline_seed1
  python -m src.train --train_csv data/augmented/train_cda.csv --run_name cda_seed0
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import config
from src.data import EmotionDataset, load_split, make_collate
from src.utils import get_device, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    per_class = f1_score(labels, preds, average=None, labels=list(range(len(config.EMOTIONS))))
    return acc, macro_f1, per_class.tolist()


def train_model(train_csv=None, run_name="baseline_seed0", seed=0,
                model_name=config.MODEL_NAME, epochs=config.EPOCHS,
                lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE):
    set_seed(seed)
    device = get_device()
    print(f"Device: {device} | model: {model_name} | run: {run_name} | seed: {seed}")

    import pandas as pd
    train_df = pd.read_csv(train_csv) if train_csv else load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(config.EMOTIONS),
        id2label=config.ID2LABEL, label2id=config.LABEL2ID).to(device)

    collate = make_collate(tokenizer.pad_token_id)
    def make_loader(df, shuffle):
        ds = EmotionDataset(df["text"], df["label"], tokenizer, config.MAX_LEN)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    test_loader = make_loader(test_df, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.06 * total_steps), total_steps)

    best_f1, best_state, history = -1.0, None, []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += out.loss.item()
        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        avg_loss = running / len(train_loader)
        history.append({"epoch": epoch, "train_loss": avg_loss,
                        "val_acc": val_acc, "val_macro_f1": val_f1})
        print(f"Epoch {epoch}: loss {avg_loss:.4f} | val acc {val_acc:.4f} | val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_f1, per_class = evaluate(model, test_loader, device)
    minutes = (time.time() - start) / 60
    print(f"\nBEST val macro-F1: {best_f1:.4f}")
    print(f"TEST accuracy: {test_acc:.4f} | TEST macro-F1: {test_f1:.4f}")
    for e, f in zip(config.EMOTIONS, per_class):
        print(f"   F1 {e:8s}: {f:.4f}")
    print(f"Training time: {minutes:.1f} min")

    model_dir = config.MODELS_DIR / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run_name": run_name, "seed": seed, "model_name": model_name,
        "train_size": len(train_df), "epochs": epochs, "lr": lr,
        "best_val_macro_f1": best_f1, "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_f1_per_class": dict(zip(config.EMOTIONS, per_class)),
        "train_minutes": minutes, "history": history,
    }
    with open(config.TABLES_DIR / f"{run_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model to {model_dir}")
    print(f"Saved metrics to results/tables/{run_name}_metrics.json")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--model_name", type=str, default=config.MODEL_NAME)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    a = p.parse_args()
    train_model(train_csv=a.train_csv,
                run_name=a.run_name or f"baseline_seed{a.seed}",
                seed=a.seed, model_name=a.model_name, epochs=a.epochs)

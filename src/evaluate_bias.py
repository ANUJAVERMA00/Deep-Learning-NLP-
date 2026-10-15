"""Phase 2 (part 2): measure affective bias of a trained model on EEC.

Usage (project root):
  python -m src.evaluate_bias --run_name baseline_seed0
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

import config
from src.metrics import compute_table
from src.utils import get_device

# female phrase -> its male counterpart (lowercase), used for gender pairing
FEMALE_TO_MALE = {
    "she": "he", "her": "him", "this woman": "this man", "this girl": "this boy",
    "my sister": "my brother", "my daughter": "my son", "my wife": "my husband",
    "my girlfriend": "my boyfriend", "my mother": "my father",
    "my aunt": "my uncle", "my mom": "my dad",
}
EXPECTED = {"gender": 1400, "race": 2800}
KEYS = ["template", "emotion word"]


def load_eec() -> pd.DataFrame:
    files = sorted((config.DATA_RAW / "eec").glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV in data/raw/eec. Run: python -m src.download_eec_hf")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["sentence", "template", "person", "gender", "race", "emotion", "emotion word"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"EEC is missing columns {miss}; found {list(df.columns)}")
    total = len(df)
    df = df.dropna(subset=["emotion", "emotion word"]).copy()
    print(f"EEC: {total} rows, kept {len(df)} with an emotion word "
          f"(dropped {total - len(df)} neutral).")

    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    unknown = set(df["emotion"]) - set(config.LABEL2ID)
    if unknown:
        raise ValueError(f"Unexpected emotion labels in EEC: {unknown}")
    df["emotion_id"] = df["emotion"].map(config.LABEL2ID)
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df["person_l"] = df["person"].astype(str).str.strip().str.lower()
    df["emotion word"] = df["emotion word"].astype(str).str.strip().str.lower()

    race = df["race"].where(df["race"].notna(), "").astype(str).str.strip().str.lower()
    df["race_key"] = race.map(lambda r: "aa" if "african" in r
                              else ("ea" if "european" in r else "none"))
    return df.reset_index(drop=True)


def _merge_pairs(a: pd.DataFrame, b: pd.DataFrame, on: list) -> pd.DataFrame:
    keep = ["sentence", "person", "emotion_id"]
    a = a[on + keep].rename(columns={c: c + "_1" for c in keep})
    b = b[on + ["sentence", "person"]].rename(
        columns={"sentence": "sentence_2", "person": "person_2"})
    return a.merge(b, on=on, how="inner")


def make_gender_pairs(df: pd.DataFrame) -> pd.DataFrame:
    nouns = df[df["race_key"] == "none"]
    male = nouns[nouns["gender"] == "male"].copy()
    female = nouns[nouns["gender"] == "female"].copy()
    female["counterpart"] = female["person_l"].map(FEMALE_TO_MALE)
    male["counterpart"] = male["person_l"]

    unmapped = sorted(female.loc[female["counterpart"].isna(), "person_l"].unique())
    if unmapped:
        print(f"WARNING: no male counterpart known for {unmapped}; "
              "falling back to pairing by order of appearance.")
        for g in (male, female):
            g["counterpart"] = g.groupby(KEYS).cumcount()
    on = KEYS + ["counterpart"]
    return _merge_pairs(male, female, on)       # g1 = male, g2 = female


def make_race_pairs(df: pd.DataFrame) -> pd.DataFrame:
    names = df[df["race_key"].isin(["ea", "aa"])]
    ea = names[names["race_key"] == "ea"].copy()
    aa = names[names["race_key"] == "aa"].copy()
    for g in (ea, aa):
        g["idx"] = g.groupby(KEYS + ["gender"]).cumcount()
    return _merge_pairs(ea, aa, KEYS + ["gender", "idx"])   # g1 = EA, g2 = AA


def check_pairs(pairs: pd.DataFrame, name: str) -> None:
    def norm(s, p):
        return re.sub(r"\b" + re.escape(p) + r"\b", "§", s, flags=re.IGNORECASE)
    ok = np.mean([norm(s1, p1) == norm(s2, p2) for s1, p1, s2, p2 in
                  zip(pairs["sentence_1"], pairs["person_1"],
                      pairs["sentence_2"], pairs["person_2"])])
    exp = EXPECTED[name]
    flag = "OK" if len(pairs) == exp else f"EXPECTED {exp}"
    print(f"[{name}] {len(pairs)} pairs ({flag}); differ only in person phrase: {ok:.1%}")
    print("   sample pairs:")
    for _, r in pairs.sample(min(5, len(pairs)), random_state=0).iterrows():
        print(f"     {r['sentence_1']!r}  <->  {r['sentence_2']!r}")


@torch.no_grad()
def predict_probs(model, tokenizer, sentences, device, batch_size=128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(list(sentences[i:i + batch_size]), padding=True, truncation=True,
                        max_length=config.MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def score_pairs(pairs: pd.DataFrame, prob_of: dict):
    p1 = np.stack([prob_of[s] for s in pairs["sentence_1"]])
    p2 = np.stack([prob_of[s] for s in pairs["sentence_2"]])
    return p1, p2


def evaluate_run(run_name: str, save: bool = True, verbose: bool = True) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_dir = config.MODELS_DIR / run_name
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} not found. Train it first (src.train).")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    df = load_eec()
    domains = {"gender": make_gender_pairs(df), "race": make_race_pairs(df)}
    for name, pairs in domains.items():
        check_pairs(pairs, name)

    all_sents = pd.unique(np.concatenate([
        np.concatenate([p["sentence_1"].values, p["sentence_2"].values])
        for p in domains.values()]))
    probs = predict_probs(model, tokenizer, all_sents, device)
    prob_of = {s: probs[i] for i, s in enumerate(all_sents)}

    # accuracy on EEC's own emotion labels (domain-shift check)
    eec_probs = predict_probs(model, tokenizer, df["sentence"].tolist(), device)
    eec_acc = float((eec_probs.argmax(1) == df["emotion_id"].values).mean())
    print(f"\nModel accuracy on EEC sentences (vs EEC emotion labels): {eec_acc:.3f}")

    results = {}
    for name, pairs in domains.items():
        p1, p2 = score_pairs(pairs, prob_of)
        table = compute_table(p1, p2, pairs["emotion_id_1"].values)
        results[name] = table
        if verbose:
            g = "M x F" if name == "gender" else "EA x AA"
            print(f"\n=== {name.upper()} ({g}) | run: {run_name} ===")
            print(table.round(4).to_string(index=False))
        if save:
            config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
            table.to_csv(config.TABLES_DIR / f"bias_{run_name}_{name}.csv", index=False)
            per = pairs[["sentence_1", "sentence_2", "emotion word"]].copy()
            for k, e in enumerate(config.EMOTIONS):
                per[f"{e}_g1"], per[f"{e}_g2"] = p1[:, k], p2[:, k]
            per["emotion_id"] = pairs["emotion_id_1"].values
            per.to_csv(config.TABLES_DIR / f"pairs_{run_name}_{name}.csv", index=False)
    if save:
        print(f"\nSaved tables to results/tables/ (bias_{run_name}_*.csv, pairs_{run_name}_*.csv)")
    results["eec_accuracy"] = eec_acc
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="baseline_seed0")
    evaluate_run(ap.parse_args().run_name)

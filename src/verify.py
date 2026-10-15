"""One-command health check for the whole project.

Run from the project root:   python -m src.verify
Options:  --runs baseline_seed0 cda_seed0     (default: every model in models/)
          --skip_model                        (skip the model-prediction sanity test)
Prints PASS / WARN / FAIL for every check and a summary at the end.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src.cda import swap_gender
from src.metrics import compute_table

STATUS = []


def check(name, ok, detail="", warn=False):
    s = "PASS" if ok else ("WARN" if warn else "FAIL")
    STATUS.append(s)
    print(f"[{s}] {name}" + (f"   ({detail})" if detail else ""))


def section(t):
    print(f"\n=== {t} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--skip_model", action="store_true")
    a = ap.parse_args()
    runs = a.runs or sorted(p.parent.name for p in config.MODELS_DIR.glob("*/config.json"))
    print("Runs found:", runs if runs else "none")

    # ------------------------------------------------------------ files
    section("1. Files")
    for name, p in {"processed train": config.DATA_PROCESSED / "train.csv",
                    "processed val": config.DATA_PROCESSED / "val.csv",
                    "processed test": config.DATA_PROCESSED / "test.csv",
                    "CDA training set": config.DATA_AUGMENTED / "train_cda.csv"}.items():
        check(f"{name} exists", p.exists(), str(p.relative_to(config.ROOT)))
    eec_files = sorted((config.DATA_RAW / "eec").glob("*.csv"))
    check("EEC csv exists", len(eec_files) > 0)
    for r in runs:
        need = [config.TABLES_DIR / f"{r}_metrics.json",
                config.TABLES_DIR / f"bias_{r}_gender.csv", config.TABLES_DIR / f"bias_{r}_race.csv",
                config.TABLES_DIR / f"pairs_{r}_gender.csv", config.TABLES_DIR / f"pairs_{r}_race.csv"]
        miss = [p.name for p in need if not p.exists()]
        check(f"{r}: metrics and bias tables exist", not miss, f"missing: {miss}" if miss else "")

    # ------------------------------------------------------------ metrics code
    section("2. Metrics code (synthetic test)")
    rng = np.random.default_rng(0)
    base = rng.dirichlet([1, 1, 1, 1], 300)
    t0 = compute_table(base, base.copy(), rng.integers(0, 4, 300))
    check("identical outputs -> DP=1, avg.delta=0, p=1",
          bool((t0.dp == 1).all() and (t0.avg_delta == 0).all() and (t0.t_p == 1).all()))
    biased = base.copy(); biased[:, 0] += 0.2; biased /= biased.sum(1, keepdims=True)
    t1 = compute_table(base, biased, None)
    ar = t1[t1.emotion == "anger"].iloc[0]
    check("planted anger bias is detected (DP<0.8, ACS>0, p<0.001)",
          bool(ar.dp < 0.8 and ar.acs > 0 and ar.t_p < 1e-3))

    # ------------------------------------------------------------ models
    section("3. Trained models")
    tok_cache = {}
    for r in runs:
        cfgp = config.MODELS_DIR / r / "config.json"
        try:
            id2 = {int(k): v for k, v in json.load(open(cfgp))["id2label"].items()}
            check(f"{r}: label order = anger, fear, joy, sadness", id2 == config.ID2LABEL, str(id2))
        except Exception as e:
            check(f"{r}: label order readable", False, str(e))
        mp = config.TABLES_DIR / f"{r}_metrics.json"
        if mp.exists():
            m = json.load(open(mp))
            check(f"{r}: test accuracy > 0.90 and macro-F1 > 0.90",
                  m["test_accuracy"] > 0.90 and m["test_macro_f1"] > 0.90,
                  f"acc {m['test_accuracy']:.4f}, macro-F1 {m['test_macro_f1']:.4f}")
    if not a.skip_model:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            sents = [("i feel so happy and grateful today", "joy"),
                     ("i am terrified of what might happen", "fear"),
                     ("i am furious at how they treated me", "anger"),
                     ("i feel so lonely and sad", "sadness")]
            for r in runs:
                d = config.MODELS_DIR / r
                tok = AutoTokenizer.from_pretrained(d)
                mod = AutoModelForSequenceClassification.from_pretrained(d).eval()
                ok = 0
                for s, want in sents:
                    with torch.no_grad():
                        p = mod(**tok(s, return_tensors="pt")).logits.softmax(-1)[0]
                    ok += mod.config.id2label[int(p.argmax())] == want
                check(f"{r}: predicts obvious emotions correctly", ok >= 3, f"{ok}/4 correct")
        except Exception as e:
            check("model prediction test could run", False, str(e)[:120], warn=True)

    # ------------------------------------------------------------ EEC pairs
    section("4. EEC sentence pairs")
    try:
        from src import evaluate_bias as eb
        df = eb.load_eec()
        g, r_ = eb.make_gender_pairs(df), eb.make_race_pairs(df)
        check("gender pairs = 1400", len(g) == 1400, str(len(g)))
        check("race pairs = 2800", len(r_) == 2800, str(len(r_)))
    except Exception as e:
        check("EEC pairs built", False, str(e)[:120])

    # ------------------------------------------------------------ CDA
    section("5. CDA data")
    pa, po = config.DATA_AUGMENTED / "train_cda.csv", config.DATA_PROCESSED / "train.csv"
    if pa.exists() and po.exists():
        o, c = pd.read_csv(po), pd.read_csv(pa)
        aug, orig = c[c.is_augmented == 1], c[c.is_augmented == 0]
        check("original rows untouched", len(orig) == len(o) and bool((orig.text.values == o.text.values).all()))
        n_sw = sum(swap_gender(t)[1] > 0 for t in o.text)
        check("augmented rows = sentences containing gendered words", len(aug) == n_sw, f"{len(aug)} vs {n_sw}")
        check("no missing text", int(c.text.isna().sum()) == 0)
        rt = sum(swap_gender(swap_gender(t)[0])[0] == t for t in o.text[:3000])
        check("swap -> swap back restores the sentence (>=95%)", rt >= 0.95 * min(3000, len(o)),
              f"{rt}/{min(3000, len(o))}")
        same = (aug.text.values == o.loc[o.text.map(lambda t: swap_gender(t)[1] > 0), "text"].values).sum()
        check("swapped sentences differ from originals", same == 0, f"{same} unchanged")
        pr_o = o.emotion.value_counts(normalize=True); pr_a = aug.emotion.value_counts(normalize=True)
        gap = max(abs(pr_o.get(e, 0) - pr_a.get(e, 0)) for e in config.EMOTIONS)
        check("augmented rows keep similar emotion mix", gap < 0.15, f"max share gap {gap:.3f}", warn=True)
        if eec_files:
            tr = set(c.text.str.lower().str.strip())
            eec = pd.concat([pd.read_csv(f) for f in eec_files])
            leak = int(eec.sentence.str.lower().str.strip().isin(tr).sum())
            check("no EEC sentence in the training data", leak == 0, f"{leak} found")
        cr = [r for r in runs if r.startswith("cda")]
        for r in cr:
            mp = config.TABLES_DIR / f"{r}_metrics.json"
            if mp.exists():
                check(f"{r}: trained on the augmented set", json.load(open(mp))["train_size"] == len(c),
                      f"train_size {json.load(open(mp))['train_size']} vs {len(c)}")

    # ------------------------------------------------------------ tables vs raw
    section("6. Bias tables vs raw per-pair scores")
    for r in runs:
        for dom in ["gender", "race"]:
            tp, pp = config.TABLES_DIR / f"bias_{r}_{dom}.csv", config.TABLES_DIR / f"pairs_{r}_{dom}.csv"
            if not (tp.exists() and pp.exists()):
                continue
            t, p = pd.read_csv(tp), pd.read_csv(pp)
            worst = 0.0
            for i, e in enumerate(config.EMOTIONS):
                s = p[p.emotion_id == i]
                d = (s[f"{e}_g1"] - s[f"{e}_g2"]).abs().mean()
                worst = max(worst, abs(d - t.loc[t.emotion == e, "avg_delta"].iloc[0]))
            check(f"{r} {dom}: avg.delta in table = recomputed", worst < 5e-4, f"max diff {worst:.1e}")

    # ------------------------------------------------------------ comparisons
    section("7. Baseline vs CDA sanity")
    seeds = sorted({r.split("seed")[1] for r in runs if "seed" in r})
    for sd in seeds:
        b, c_ = f"baseline_seed{sd}", f"cda_seed{sd}"
        mb, mc = config.TABLES_DIR / f"{b}_metrics.json", config.TABLES_DIR / f"{c_}_metrics.json"
        if mb.exists() and mc.exists():
            jb, jc = json.load(open(mb)), json.load(open(mc))
            d = jc["test_macro_f1"] - jb["test_macro_f1"]
            check(f"seed {sd}: CDA macro-F1 within 0.02 of baseline", abs(d) <= 0.02,
                  f"difference {d:+.4f}", warn=True)
            check(f"seed {sd}: CDA trained on more data", jc["train_size"] > jb["train_size"],
                  f"{jc['train_size']} vs {jb['train_size']}")
            tb, tc = config.TABLES_DIR / f"bias_{b}_gender.csv", config.TABLES_DIR / f"bias_{c_}_gender.csv"
            if tb.exists() and tc.exists():
                same = np.allclose(pd.read_csv(tb).avg_delta, pd.read_csv(tc).avg_delta)
                check(f"seed {sd}: baseline and CDA bias tables differ", not same)

    n = {k: STATUS.count(k) for k in ("PASS", "WARN", "FAIL")}
    print(f"\nSUMMARY: {n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed")
    print("All good." if n["FAIL"] == 0 else "Fix the FAIL items before continuing.")


if __name__ == "__main__":
    main()

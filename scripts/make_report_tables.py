"""Phase 4: aggregate all finished runs into mean ± std tables.

Usage (project root):  python -m scripts.make_report_tables
Writes CSVs and results/tables/summary.md
"""
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

import config

DOMAINS = ["gender", "race"]
LABEL = {"baseline": "Baseline", "cda": "+ CDA"}


def load_runs():
    """[(condition, seed, run_name)] for every finished run."""
    out = []
    for f in sorted(config.TABLES_DIR.glob("*_metrics.json")):
        m = re.match(r"(baseline|cda)_seed(\d+)_metrics\.json$", f.name)
        if not m:
            continue
        run = f"{m.group(1)}_seed{m.group(2)}"
        if all((config.TABLES_DIR / f"bias_{run}_{d}.csv").exists() for d in DOMAINS):
            out.append((m.group(1), int(m.group(2)), run))
    return out


def ms(values, nd=4):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def pair_gaps(run, domain):
    """Per-pair |p(e,g1) - p(e,g2)| using each pair's own emotion. Returns (gaps, emotion_ids, key)."""
    p = pd.read_csv(config.TABLES_DIR / f"pairs_{run}_{domain}.csv")
    n = len(p)
    g1 = p[[f"{e}_g1" for e in config.EMOTIONS]].values[np.arange(n), p["emotion_id"].values]
    g2 = p[[f"{e}_g2" for e in config.EMOTIONS]].values[np.arange(n), p["emotion_id"].values]
    key = (p["sentence_1"] + " || " + p["sentence_2"]).values
    return np.abs(g1 - g2), p["emotion_id"].values, key


def domain_gaps(runs, domain):
    """Seed-averaged per-pair gaps per condition (pairs must be identical across runs)."""
    res, ref_key = {}, None
    for cond in ("baseline", "cda"):
        arrs = []
        for c, s, r in runs:
            if c != cond:
                continue
            g, e, k = pair_gaps(r, domain)
            if ref_key is None:
                ref_key, ref_e = k, e
            assert (k == ref_key).all(), f"pair order differs for {r}"
            arrs.append(g)
        if arrs:
            res[cond] = np.mean(arrs, axis=0)
    return res, ref_e if ref_key is not None else None


def main():
    runs = load_runs()
    if not runs:
        sys.exit("No finished runs found in results/tables/")
    conds = [c for c in ("baseline", "cda") if any(r[0] == c for r in runs)]
    seeds = {c: sorted(s for cc, s, _ in runs if cc == c) for c in conds}
    md = ["# Results summary", "",
          "Seeds used: " + "; ".join(f"{LABEL[c]}: {seeds[c]}" for c in conds), "",
          "Values are mean ± standard deviation over seeds (single value if only one seed)."]
    print("Seeds:", seeds)

    # ---------------- utility
    rows = []
    import json
    for c in conds:
        ms_ = [json.load(open(config.TABLES_DIR / f"{r}_metrics.json")) for cc, s, r in runs if cc == c]
        row = {"condition": LABEL[c], "seeds": len(ms_),
               "train size": ms_[0]["train_size"],
               "test accuracy": ms([m["test_accuracy"] for m in ms_]),
               "test macro-F1": ms([m["test_macro_f1"] for m in ms_])}
        for e in config.EMOTIONS:
            row[f"F1 {e}"] = ms([m["test_f1_per_class"][e] for m in ms_], 3)
        rows.append(row)
    util = pd.DataFrame(rows)
    util.to_csv(config.TABLES_DIR / "summary_utility.csv", index=False)
    md += ["", "## 1. Classification quality (test set)", "", md_table(util)]
    print("\nUTILITY\n", util.to_string(index=False))

    # ---------------- per-emotion bias
    for dom in DOMAINS:
        rows = []
        for e in config.EMOTIONS:
            for c in conds:
                ts = [pd.read_csv(config.TABLES_DIR / f"bias_{r}_{dom}.csv").set_index("emotion").loc[e]
                      for cc, s, r in runs if cc == c]
                n = len(ts)
                rows.append({"emotion": e, "condition": LABEL[c], "seeds": n,
                             "DP": ms([t["dp"] for t in ts], 3),
                             "avg.Δ": ms([t["avg_delta"] for t in ts]),
                             "ACS": ms([t["acs"] for t in ts]),
                             "t-test sig (Holm)": f"{sum(t['t_p_holm'] < config.ALPHA for t in ts)}/{n}",
                             "Wilcoxon sig (Holm)": f"{sum(t['wilcoxon_p_holm'] < config.ALPHA for t in ts)}/{n}"})
        df = pd.DataFrame(rows)
        df.to_csv(config.TABLES_DIR / f"summary_bias_{dom}.csv", index=False)
        name = "Gender (male × female)" if dom == "gender" else "Race (EA × AA)"
        md += ["", f"## 2. Bias per emotion: {name}", "",
               "'sig' = number of seeds with Holm-corrected p < 0.05.", "", md_table(df)]
        print(f"\nBIAS {dom.upper()}\n", df.to_string(index=False))

    # ---------------- overall + paired comparison
    over, paired = [], []
    for dom in DOMAINS:
        per = {}
        for c in conds:
            per[c] = []
            for cc, s, r in runs:
                if cc == c:
                    t = pd.read_csv(config.TABLES_DIR / f"bias_{r}_{dom}.csv")
                    per[c].append((t["avg_delta"].mean(), t["dp"].min()))
        for c in conds:
            vals = np.array(per[c])
            over.append({"domain": dom, "condition": LABEL[c], "seeds": len(vals),
                         "mean avg.Δ over emotions": ms(vals[:, 0], 5), "lowest DP": ms(vals[:, 1], 3)})
        if "baseline" in per and "cda" in per:
            b, d = np.array(per["baseline"])[:, 0], np.array(per["cda"])[:, 0]
            diff = d.mean() - b.mean()
            noise = max(b.std(ddof=1) if len(b) > 1 else 0, d.std(ddof=1) if len(d) > 1 else 0)
            over[-1]["change vs baseline"] = f"{diff:+.5f} ({100 * diff / b.mean():+.1f}%)"
            if noise > 0:
                over[-1]["beyond seed noise?"] = ("no (inconclusive)" if abs(diff) <= noise else
                                                  ("yes: CDA lower bias" if diff < 0 else "yes: CDA HIGHER bias"))
            else:
                over[-1]["beyond seed noise?"] = "n/a (1 seed)"
            gaps, _ = domain_gaps(runs, dom)
            gb, gc = gaps["baseline"], gaps["cda"]
            try:
                p = stats.wilcoxon(gb, gc).pvalue
            except ValueError:
                p = 1.0
            paired.append({"domain": dom, "pairs": len(gb),
                           "mean gap baseline": f"{gb.mean():.5f}", "mean gap CDA": f"{gc.mean():.5f}",
                           "relative change": f"{100 * (gc.mean() - gb.mean()) / gb.mean():+.1f}%",
                           "pairs with smaller gap under CDA": f"{100 * np.mean(gc < gb):.1f}%",
                           "Wilcoxon p (paired over pairs)": f"{p:.2e}"})
    over_df = pd.DataFrame(over).fillna("")
    over_df.to_csv(config.TABLES_DIR / "summary_overall.csv", index=False)
    md += ["", "## 3. Overall bias (mean avg.Δ over the four emotions)", "",
           "'beyond seed noise?' compares the baseline-vs-CDA difference with the seed-to-seed standard deviation (a rough rule of thumb).",
           "", md_table(over_df)]
    print("\nOVERALL\n", over_df.to_string(index=False))
    if paired:
        pdf = pd.DataFrame(paired)
        pdf.to_csv(config.TABLES_DIR / "summary_paired.csv", index=False)
        md += ["", "## 4. Paired comparison over sentence pairs (seed-averaged)", "",
               "Each pair’s absolute score gap is averaged over seeds, then baseline and CDA are compared pair by pair. "
               "Pairs share templates, so they are not fully independent: treat the p-value as indicative.", "", md_table(pdf)]
        print("\nPAIRED\n", pdf.to_string(index=False))

    (config.TABLES_DIR / "summary.md").write_text("\n".join(md) + "\n")
    print("\nSaved results/tables/summary.md and summary_*.csv")


if __name__ == "__main__":
    main()

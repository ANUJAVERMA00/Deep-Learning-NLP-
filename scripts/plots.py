"""Phase 4: figures. Usage (project root):  python -m scripts.plots"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from scripts.make_report_tables import LABEL, domain_gaps, load_runs

COL = {"baseline": "#1F3A5F", "cda": "#E07B39"}


def main():
    runs = load_runs()
    if not runs:
        sys.exit("No finished runs found.")
    conds = [c for c in ("baseline", "cda") if any(r[0] == c for r in runs)]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- fig 1: utility
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    w = 0.35
    for j, c in enumerate(conds):
        ms_ = [json.load(open(config.TABLES_DIR / f"{r}_metrics.json")) for cc, s, r in runs if cc == c]
        vals = [[m["test_accuracy"] for m in ms_], [m["test_macro_f1"] for m in ms_]]
        mean = [np.mean(v) for v in vals]
        err = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals]
        ax.bar(np.arange(2) + (j - 0.5) * w, mean, w, yerr=err, capsize=3, color=COL[c], label=LABEL[c])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Accuracy", "Macro-F1"])
    ax.set_ylim(0.90, 1.0); ax.set_ylabel("Test score (axis starts at 0.90)")
    ax.set_title("Classification quality"); ax.legend(loc="lower center", frameon=False)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "fig1_utility.png", dpi=160); plt.close(fig)

    # ---- fig 2: avg.delta per emotion
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, dom, title in zip(axes, ["gender", "race"], ["Gender (male × female)", "Race (EA × AA)"]):
        for j, c in enumerate(conds):
            tabs = [pd.read_csv(config.TABLES_DIR / f"bias_{r}_{dom}.csv").set_index("emotion")
                    for cc, s, r in runs if cc == c]
            vals = np.array([[t.loc[e, "avg_delta"] for e in config.EMOTIONS] for t in tabs])
            err = vals.std(axis=0, ddof=1) if len(vals) > 1 else np.zeros(vals.shape[1])
            ax.bar(np.arange(4) + (j - 0.5) * w, vals.mean(0), w, yerr=err, capsize=3,
                   color=COL[c], label=f"{LABEL[c]} (n={len(vals)})")
        ax.set_xticks(range(4)); ax.set_xticklabels(config.EMOTIONS)
        ax.set_ylabel("avg.Δ (mean |score gap|)"); ax.set_title(title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3); ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Intensity bias on EEC (lower is fairer; bars = mean ± std over seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "fig2_avg_delta.png", dpi=160); plt.close(fig)

    # ---- fig 3: per-pair gap distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.logspace(-5, 0, 40)
    for ax, dom, title in zip(axes, ["gender", "race"], ["Gender", "Race"]):
        gaps, _ = domain_gaps(runs, dom)
        for c in conds:
            ax.hist(np.clip(gaps[c], 1e-5, 1), bins=bins, histtype="step", linewidth=1.8,
                    color=COL[c], label=f"{LABEL[c]} (mean {gaps[c].mean():.4f})")
        ax.set_xscale("log"); ax.set_xlabel("per-pair |score gap| (seed-averaged)")
        ax.set_ylabel("number of pairs"); ax.set_title(title); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "fig3_gap_distribution.png", dpi=160); plt.close(fig)
    print("Saved fig1_utility.png, fig2_avg_delta.png, fig3_gap_distribution.png in results/figures/")


if __name__ == "__main__":
    main()

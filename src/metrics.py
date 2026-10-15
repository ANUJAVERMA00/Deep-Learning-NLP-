"""Phase 2 (part 1): bias metrics from Kadan et al. (2024), plus two fixes.

Inputs are softmax probabilities for sentence pairs that differ only in the
group term:  p1[i] = probs for the sentence about group g1,
             p2[i] = probs for the sentence about group g2,   shape (N, 4).

Metrics (computed per emotion e):
  DP         min(rate1, rate2) / max(rate1, rate2), where rate = share of
             sentences predicted as e.  1.0 = fair, < 0.80 = biased.
  avg_delta  mean |p1[:, e] - p2[:, e]|
  ACS        mean(1 - p1[:, e] / p2[:, e]).  Negative => g1 scores higher.
  t-test     paired t-test on the scores (as in the paper)
  Wilcoxon   paired signed-rank test (added: softmax scores are bounded
             and not normal, so this is the safer test)
  Holm       multiple-comparison correction across the emotions (added)

Differences from the paper: we compare the probability of the SAME emotion e
for both sentences (the paper compares the max score, which can belong to
different classes).

Self-test:  python -m src.metrics
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

import config

EPS = 1e-6


def dp_ratio(pred1: np.ndarray, pred2: np.ndarray, e: int) -> float:
    r1, r2 = float(np.mean(pred1 == e)), float(np.mean(pred2 == e))
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == 0:          # emotion never predicted for either group
        return 1.0
    return lo / hi


def _paired_pvalues(s1: np.ndarray, s2: np.ndarray):
    diff = s1 - s2
    if len(s1) < 2 or np.allclose(diff, 0):
        return 1.0, 1.0
    t_p = float(stats.ttest_rel(s1, s2).pvalue)
    try:
        w_p = float(stats.wilcoxon(s1, s2).pvalue)
    except ValueError:
        w_p = 1.0
    return t_p, w_p


def holm_correction(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bias_for_emotion(p1: np.ndarray, p2: np.ndarray, e: int) -> dict:
    s1, s2 = p1[:, e], p2[:, e]
    t_p, w_p = _paired_pvalues(s1, s2)
    return {
        "n_pairs": len(s1),
        "dp": dp_ratio(p1.argmax(1), p2.argmax(1), e),
        "avg_delta": float(np.mean(np.abs(s1 - s2))),
        "acs": float(np.mean(1.0 - s1 / np.clip(s2, EPS, None))),
        "t_p": t_p,
        "wilcoxon_p": w_p,
        "mean_g1": float(s1.mean()),
        "mean_g2": float(s2.mean()),
    }


def compute_table(p1: np.ndarray, p2: np.ndarray, pair_emotions=None) -> pd.DataFrame:
    """One row per emotion.

    pair_emotions: optional array of EEC ground-truth emotion ids (0..3) per
    pair. If given, emotion e is evaluated only on pairs whose sentence
    expresses e (as in the paper's per-emotion tables). If None (e.g. CSP),
    every pair is used for every emotion.
    """
    p1, p2 = np.asarray(p1), np.asarray(p2)
    rows = []
    for e, name in enumerate(config.EMOTIONS):
        if pair_emotions is not None:
            mask = np.asarray(pair_emotions) == e
            if mask.sum() == 0:
                continue
            row = bias_for_emotion(p1[mask], p2[mask], e)
        else:
            row = bias_for_emotion(p1, p2, e)
        row["emotion"] = name
        rows.append(row)
    df = pd.DataFrame(rows)
    df["t_p_holm"] = holm_correction(df["t_p"])
    df["wilcoxon_p_holm"] = holm_correction(df["wilcoxon_p"])
    df["dp_biased"] = df["dp"] < config.DP_THRESHOLD
    df["sig_t"] = df["t_p"] < config.ALPHA
    df["sig_wilcoxon_holm"] = df["wilcoxon_p_holm"] < config.ALPHA
    cols = ["emotion", "n_pairs", "dp", "avg_delta", "acs", "t_p", "wilcoxon_p",
            "t_p_holm", "wilcoxon_p_holm", "mean_g1", "mean_g2",
            "dp_biased", "sig_t", "sig_wilcoxon_holm"]
    return df[cols]


def _selftest():
    rng = np.random.default_rng(0)
    n = 400
    base = rng.dirichlet([1, 1, 1, 1], size=n)

    print("Case 1: identical outputs for both groups (no bias expected)")
    print(compute_table(base, base.copy(), rng.integers(0, 4, n)).round(4).to_string(index=False))

    print("\nCase 2: group 2 gets +0.15 'anger' probability (bias expected)")
    biased = base.copy()
    biased[:, 0] += 0.15
    biased /= biased.sum(1, keepdims=True)
    print(compute_table(base, biased, None).round(4).to_string(index=False))
    print("\nExpected: case 1 -> DP 1.0, avg_delta 0, p = 1.0;"
          "\n          case 2 -> anger row: ACS positive (group 2 higher), DP < 0.8, tiny p-values.")
    print("(Other emotions drop slightly because probabilities are renormalised.)")


if __name__ == "__main__":
    _selftest()

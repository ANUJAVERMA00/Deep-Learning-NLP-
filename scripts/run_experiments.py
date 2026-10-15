"""Phase 4: train + evaluate baseline and CDA for several seeds.

Usage (project root):
  caffeinate -i python -m scripts.run_experiments            # seeds 0 1 2, both conditions
  caffeinate -i python -m scripts.run_experiments --seeds 1 2
Finished runs are skipped, so you can stop (Ctrl+C) and rerun safely.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from src.evaluate_bias import evaluate_run
from src.train import train_model

CDA_CSV = config.DATA_AUGMENTED / "train_cda.csv"


def has_model(run):
    return ((config.MODELS_DIR / run / "config.json").exists()
            and (config.TABLES_DIR / f"{run}_metrics.json").exists())


def has_bias(run):
    return all((config.TABLES_DIR / f"bias_{run}_{d}.csv").exists() for d in ("gender", "race"))


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--conditions", nargs="+", default=["baseline", "cda"], choices=["baseline", "cda"])
    a = ap.parse_args()
    if "cda" in a.conditions and not CDA_CSV.exists():
        sys.exit("data/augmented/train_cda.csv not found. Run: python -m src.cda")

    jobs = [(c, s) for s in a.seeds for c in a.conditions]      # interleaved by seed
    t0 = time.time()
    for i, (cond, seed) in enumerate(jobs, 1):
        run = f"{cond}_seed{seed}"
        print(f"\n{'=' * 60}\n[{i}/{len(jobs)}] {run}\n{'=' * 60}")
        if has_model(run) and has_bias(run):
            print("already finished, skipping")
            continue
        if not has_model(run):
            train_model(train_csv=str(CDA_CSV) if cond == "cda" else None,
                        run_name=run, seed=seed)
            free_memory()
        evaluate_run(run, verbose=False)
        free_memory()
        print(f"elapsed so far: {(time.time() - t0) / 60:.1f} min")
    print("\nAll requested runs finished. Next: python -m scripts.make_report_tables")


if __name__ == "__main__":
    main()

# Affective bias in an emotion classifier: measurement and a CDA baseline

Small-scale study of **affective bias** in a pre-trained language model. A DistilBERT emotion
classifier (anger, fear, joy, sadness) is tested on sentence pairs that differ only in a
gender or race phrase, and counterfactual data augmentation (CDA) is tested as a mitigation baseline.

It extends the measurement approach of Kadan, Deepak P., Bhadra, Gangan and Lajish (2024),
*Understanding latent affective bias in large pre-trained neural language models*,
Natural Language Processing Journal 7, 100062. That paper measures the bias but tests no mitigation.

## What this project does

1. Fine-tunes `distilbert-base-uncased` on `dair-ai/emotion` filtered to four emotions.
2. Builds 1,400 gender pairs and 2,800 race pairs from the Equity Evaluation Corpus (EEC).
3. Measures demographic parity (DP), average score gap (avg.Δ), ACS, a t-test, and, added here,
   a Wilcoxon test with Holm correction, using the probability of the same emotion for both sentences.
4. Applies CDA (swaps gendered words in training sentences), retrains, and measures again.
5. Repeats baseline and CDA with 3 seeds each, with race as a control.

## Main results (mean ± std over 3 seeds)

| | Baseline | + CDA |
|---|---|---|
| Test accuracy | 0.9679 ± 0.0028 | 0.9636 ± 0.0027 |
| Test macro-F1 | 0.9590 ± 0.0037 | 0.9530 ± 0.0035 |
| Gender: mean avg.Δ over emotions | 0.00625 ± 0.00050 | 0.00604 ± 0.00057 (−3.4%) |
| Race (control): mean avg.Δ | 0.01249 ± 0.00178 | 0.01234 ± 0.00242 (−1.2%) |

- Class-level bias is almost absent (DP of 0.98 or higher), but small score gaps (about 0.3 to 1.5 percentage points) are detectable, as in the paper.
- CDA did **not** measurably reduce gender bias: the change is inside seed-to-seed noise, the race control moved too, and macro-F1 fell by about 0.6 points.
- A paired test over sentence pairs looked extremely significant but also held for race, and the apparent improvement on the worst pairs was regression to the mean. See the paper in `docs/` for details.

## Repository layout

```
config.py                     settings (model, learning rate, epochs, seeds, paths)
requirements.txt              packages
src/
  data.py                     filters dair-ai/emotion to four classes, dataset, batching
  train.py                    fine-tuning, validation, test metrics
  metrics.py                  DP, avg.Δ, ACS, t-test, Wilcoxon test, Holm correction
  evaluate_bias.py            builds EEC pairs, scores a model, writes bias tables
  cda.py                      swap dictionary and augmented training set
  verify.py                   64 automatic checks
  download_data.py            downloads the datasets
  download_eec_hf.py          fetches EEC from the Hugging Face copy
  setup_check.py              checks packages, device and data files
  utils.py                    device selection and seeding
scripts/
  run_experiments.py          trains and evaluates baseline and CDA for several seeds
  make_report_tables.py       aggregates mean ± std tables into results/tables/summary.md
  plots.py                    figures into results/figures/
results/                      tables and figures from the runs reported above
docs/                         paper and report
```

Trained model checkpoints (`models/`) and the downloaded data are not stored in the repository;
the commands below recreate them.

## Setup

Python 3.12 is recommended (Python 3.14 was too new for the packages when this was developed).
Developed on a MacBook Air with an Apple M4 chip (PyTorch MPS backend). The code falls back to
CUDA or CPU if MPS is unavailable; those paths were not tested.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_data       # emotion data; the EEC zip step may print a failure message
python -m src.download_eec_hf     # fetches EEC from Hugging Face
python -m src.setup_check
```

## Reproduce the results

```bash
python -m src.data                      # filtered train/val/test files
python -m src.cda                       # augmented training set
python -m scripts.run_experiments       # 3 seeds x (baseline, CDA); about 80 minutes on an M4
python -m scripts.make_report_tables    # results/tables/summary.md
python -m scripts.plots                 # results/figures/
python -m src.verify                    # expect: 64 passed, 0 warnings, 0 failed
```

Training is not guaranteed to be bit-for-bit identical across machines or on the MPS backend
(not tested), so small numerical differences are possible.

## Technical stack

Python 3.12, PyTorch 2.14.1, Hugging Face Transformers 5.18.0, Hugging Face datasets 5.0.1,
SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, NumPy, matplotlib, tqdm.

## Limitations

Synthetic template test sentences (EEC), accuracy on EEC lower than on the test set, three seeds,
rule-based gender swaps, binary gender and two racial groups only, one small model (66M parameters)
and one training dataset. Explanations for why CDA did not help are hypotheses and were not tested.

## Reference

Kadan, A., Deepak P., Bhadra, S., Gangan, M. P., and Lajish V. L. (2024). Understanding latent affective bias
in large pre-trained neural language models. *Natural Language Processing Journal*, 7, 100062.

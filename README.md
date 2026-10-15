# Detecting and Mitigating Affective Bias in a Fine-tuned Emotion Classifier

Extension of Kadan et al. (2024), "Understanding latent affective bias in large
pre-trained neural language models". We measure affective bias in a fine-tuned
emotion classifier using the paper's metrics and mitigate it with
Counterfactual Data Augmentation (CDA).

## Phase 0: Setup (macOS, Apple Silicon)

```bash
cd affective-bias-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_data     # downloads emotion data, EEC, CrowS-Pairs
python -m src.setup_check       # verifies everything
```

Expected: device is `mps`, all data files say OK.

## Phases
0 Setup | 1 Baseline model | 2 Bias evaluation | 3 CDA | 4 Multi-seed comparison | 5 Extras | 6 Report

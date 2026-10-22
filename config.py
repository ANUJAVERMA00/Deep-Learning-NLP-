
"""Central configuration. Change things here, not inside other scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Paths ----
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_AUGMENTED = ROOT / "data" / "augmented"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

# ---- Model / training ----
MODEL_NAME = "distilbert-base-uncased"   # later: "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 4
SEEDS = [0, 1, 2]

# ---- Emotions (same 4 as the paper) ----
EMOTIONS = ["anger", "fear", "joy", "sadness"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}

# dair-ai/emotion original label ids -> names
DAIR_ID2NAME = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

# ---- Bias evaluation ----
DP_THRESHOLD = 0.80
ALPHA = 0.05


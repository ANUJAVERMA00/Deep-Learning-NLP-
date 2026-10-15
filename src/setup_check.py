"""Phase 0 checkpoint: verifies environment, device and data.

Run from the project root:  python -m src.setup_check
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.utils import get_device


def ok(flag):
    return "OK     " if flag else "MISSING"


def main():
    import torch, transformers, datasets, scipy, sklearn, pandas
    print("Python      :", sys.version.split()[0])
    print("torch       :", torch.__version__)
    print("transformers:", transformers.__version__)
    print("datasets    :", datasets.__version__)
    print("scipy       :", scipy.__version__)
    print("sklearn     :", sklearn.__version__)
    print("pandas      :", pandas.__version__)

    device = get_device()
    print("\nDevice      :", device)
    x = torch.randn(256, 256, device=device)
    y = (x @ x).sum().item()
    print("Tensor math on device works:", round(y, 2))

    print("\nData files:")
    checks = {
        "emotion_train.csv": config.DATA_RAW / "emotion_train.csv",
        "emotion_validation.csv": config.DATA_RAW / "emotion_validation.csv",
        "emotion_test.csv": config.DATA_RAW / "emotion_test.csv",
        "EEC folder": config.DATA_RAW / "eec",
        "crows_pairs.csv (optional)": config.DATA_RAW / "crows_pairs.csv",
    }
    for name, p in checks.items():
        print(f"  [{ok(p.exists())}] {name}")

    eec = list((config.DATA_RAW / "eec").rglob("*.csv"))
    if eec:
        import pandas as pd
        df = pd.read_csv(eec[0])
        print(f"\nEEC file: {eec[0].name}, {len(df)} rows, columns: {list(df.columns)}")

    if device.type == "cpu":
        print("\nNote: no GPU found, training will be slow. Check your PyTorch install.")
    print("\nPhase 0 complete if everything above says OK.")


if __name__ == "__main__":
    main()

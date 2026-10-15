"""Phase 0: download all datasets into data/raw/.

Run from the project root:  python -m src.download_data
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

EEC_URL = "https://saifmohammad.com/WebDocs/EEC/Equity-Evaluation-Corpus.zip"
CROWS_URL = ("https://raw.githubusercontent.com/nyu-mll/crows-pairs/"
             "master/data/crows_pairs_anonymized.csv")


def download_emotion():
    from datasets import load_dataset
    print("[1/3] Downloading dair-ai/emotion from Hugging Face ...")
    ds = load_dataset("dair-ai/emotion")
    for split in ds:
        out = config.DATA_RAW / f"emotion_{split}.csv"
        ds[split].to_pandas().to_csv(out, index=False)
        print(f"   saved {out.name}: {len(ds[split])} rows")


def download_eec():
    print("[2/3] Downloading Equity Evaluation Corpus (EEC) ...")
    try:
        with urllib.request.urlopen(EEC_URL, timeout=60) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(config.DATA_RAW / "eec")
        print("   extracted to data/raw/eec/")
    except Exception as e:
        print(f"   FAILED ({e}).\n   Manual fix: download the zip from "
              "https://saifmohammad.com/WebDocs/EEC/Equity-Evaluation-Corpus.zip\n"
              "   and unzip it into data/raw/eec/")


def download_crows():
    print("[3/3] Downloading CrowS-Pairs (optional, used later for religion) ...")
    try:
        out = config.DATA_RAW / "crows_pairs.csv"
        urllib.request.urlretrieve(CROWS_URL, out)
        print(f"   saved {out.name}")
    except Exception as e:
        print(f"   FAILED ({e}). You can skip this for now.")


if __name__ == "__main__":
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    download_emotion()
    download_eec()
    download_crows()
    print("\nDone. Now run:  python -m src.setup_check")

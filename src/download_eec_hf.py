import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "peixian/equity_evaluation_corpus"
REV = "refs/convert/parquet"

files = [f for f in list_repo_files(REPO, repo_type="dataset", revision=REV)
         if f.endswith(".parquet")]
print("Parquet files found:", files)

out_dir = config.DATA_RAW / "eec"
out_dir.mkdir(parents=True, exist_ok=True)

for f in files:
    local = hf_hub_download(REPO, f, repo_type="dataset", revision=REV)
    df = pd.read_parquet(local)
    name = f.replace("/", "_").replace(".parquet", ".csv")
    df.to_csv(out_dir / name, index=False)
    print(f"\nSaved {name}: {len(df)} rows")
    print("Columns:", list(df.columns))
    print(df.head(3).to_string())

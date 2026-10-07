from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from .modeling import prepare_records, score_records
except ImportError:  # Allows direct execution from the repository root.
    from modeling import prepare_records, score_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    args = parser.parse_args()

    bundle = joblib.load(args.artifact)
    raw = pd.read_csv(args.input)
    prepared = prepare_records(raw, bundle["partners"], bundle["products"])
    scores = score_records(bundle, prepared)
    pd.DataFrame({"claim_id": prepared["claim_id"], "score": scores}).to_csv(
        args.output, index=False
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()


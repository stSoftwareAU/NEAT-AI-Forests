#!/usr/bin/env python3
"""XGBoost external control (Issue #13).

Trains deliberately shallow gradient-boosted trees on the incumbent's
correction-space residuals exported by `neat_ai_forests export-matrix`, and
writes the booster's JSON dump for `neat_ai_forests import-xgboost`.

    pip install xgboost pandas
    neat_ai_forests creature.json training/ export-matrix --out matrix.csv
    scripts/xgboost-control.py matrix.csv --depth 1 --rounds 8 --out dump.json
    neat_ai_forests creature.json training/ import-xgboost --dump dump.json --scorer rust_scorer

The XGBoost training metric is recorded for context only; every converted
tree is judged by the full-corpus NEAT-AI-scorer exactly like a native
candidate. `base_score=0` so each tree is a pure additive correction.
"""
import argparse
import json
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("matrix", help="CSV from `neat_ai_forests export-matrix`")
    p.add_argument("--depth", type=int, default=1, help="max tree depth (start shallow: 1-3)")
    p.add_argument("--rounds", type=int, default=8, help="boosting rounds (= trees dumped)")
    p.add_argument("--eta", type=float, default=0.3, help="learning rate")
    p.add_argument("--min-child-weight", type=float, default=50.0, help="minimum records per leaf")
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--colsample-bytree", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="xgboost-dump.json")
    args = p.parse_args()

    try:
        import pandas as pd  # noqa: WPS433
        import xgboost as xgb  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"missing dependency: {exc}. Install with: pip install xgboost pandas", file=sys.stderr)
        return 2

    started = time.time()
    frame = pd.read_csv(args.matrix)
    features = [c for c in frame.columns if c.startswith("f")]
    target = frame["correction"].astype("float32")
    matrix = xgb.DMatrix(frame[features].astype("float32"), label=target, feature_names=features)
    params = {
        "objective": "reg:squarederror",
        "base_score": 0.0,
        "max_depth": args.depth,
        "eta": args.eta,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "seed": args.seed,
        "tree_method": "hist",
        "max_bin": 256,
    }
    evals = {}
    booster = xgb.train(params, matrix, num_boost_round=args.rounds, evals=[(matrix, "train")], evals_result=evals, verbose_eval=False)
    dump = [json.loads(t) for t in booster.get_dump(dump_format="json")]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dump, fh)
    meta = {
        "matrix": args.matrix,
        "records": int(len(frame)),
        "features": len(features),
        "params": params,
        "rounds": args.rounds,
        "trainRmse": evals.get("train", {}).get("rmse"),
        "trainSeconds": time.time() - started,
        "xgboostVersion": xgb.__version__,
        "note": "training metric is context only; NEAT-AI-scorer decides acceptance",
    }
    with open(args.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

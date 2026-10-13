"""Reproduce all results and figures.

Usage:
    python run_pipeline.py --data tweets_preprocessed_9582.csv --out outputs
"""

import argparse

import matplotlib

matplotlib.use("Agg")

from dwt_semantics.config import Config  # noqa: E402
from dwt_semantics.pipeline import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to tweets_preprocessed_9582.csv")
    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument("--bin-days", type=int, default=Config.bin_days)
    parser.add_argument("--wavelet", default=Config.wavelet)
    parser.add_argument("--model", default=Config.model_name, help="sentence-transformers model")
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()

    cfg = Config(bin_days=args.bin_days, wavelet=args.wavelet, model_name=args.model, seed=args.seed)
    out = run(args.data, args.out, cfg)
    print(out["attrition"].to_string(index=False))
    print()
    print(out["statistics"].to_string(index=False))
    print(f"\nResults written to {args.out}/")


if __name__ == "__main__":
    main()

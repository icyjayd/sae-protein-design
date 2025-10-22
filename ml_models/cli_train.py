import argparse, json
from .train import train_model


def main():
    ap = argparse.ArgumentParser(description="Train ML models for protein sequence scoring")

    # Inputs
    ap.add_argument("sequences", help="Path to sequences file (.npy or .csv)")
    ap.add_argument("--labels", help="Optional labels file (.npy or .csv)")
    
    # Data sampling
    ap.add_argument("--n-samples", type=int, default=None, 
                    help="Number of samples to use from dataset (default: use all)")

    # Optional split reuse/storage
    ap.add_argument("--split", help="Path to JSON file with saved train/test indices for reuse")
    ap.add_argument("--no-save-split", action="store_true", help="Do not save split indices to outdir")
    ap.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for test split (0-1)")
    ap.add_argument(
        "--stratify",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Stratify split: auto (default: classification only), yes, or no",
    )

    # Model/encoding
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--encoding", default="onehot", choices=["aac", "kmer", "onehot", "esm"])
    ap.add_argument("--task", choices=["classification", "regression"])
    ap.add_argument("--seed", type=int, default=42)

    # Output and encoding knobs
    ap.add_argument("--outdir", default="runs/ml_model")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=512)

    a = ap.parse_args()

    res = train_model(
        seq_file=a.sequences,
        labels_file=a.labels,
        model_name=a.model,
        encoding=a.encoding,
        task=a.task,
        seed=a.seed,
        outdir=a.outdir,
        split_file=a.split,
        save_split=(not a.no_save_split),
        test_size=a.test_size,
        stratify=a.stratify,
        k=a.k,
        max_len=a.max_len,
        n_samples=a.n_samples,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
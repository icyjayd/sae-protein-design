import argparse
from .train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train protein ML model")

    parser.add_argument("sequences", type=str, help="Path to sequence CSV or NPY file")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels CSV or NPY file")
    parser.add_argument("--task", type=str, default=None, choices=["regression", "classification", "auto"])
    parser.add_argument("--model", type=str, default="rf", help="Model name")
    parser.add_argument("--encoding", type=str, default="aac", help="Encoding: aac, kmer, onehot, esm")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples to use (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--stratify", type=str, default="auto", choices=["auto", "none"])
    parser.add_argument("--k", type=int, default=3, help="k for k-mer encoding")
    parser.add_argument("--max-len", type=int, default=512, help="Max length for one-hot encoding")
    parser.add_argument("--split", type=str, default=None, help="Path to split JSON to reuse")
    parser.add_argument("--no-save-split", action="store_true", help="Disable saving split indices")

    # new arguments
    parser.add_argument("--outdir", type=str, default=None, help="Explicit output directory (optional)")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix for auto-generated folder name")

    args = parser.parse_args()

    train_model(
        seq_file=args.sequences,
        labels_file=args.labels,
        task=None if args.task == "auto" else args.task,
        model_name=args.model,
        encoding=args.encoding,
        seed=args.seed,
        outdir=args.outdir,
        prefix=args.prefix,
        n_samples=args.n_samples,
        test_size=args.test_size,
        stratify=args.stratify,
        k=args.k,
        max_len=args.max_len,
        split=args.split,
        save_split=not args.no_save_split,
    )


if __name__ == "__main__":
    main()

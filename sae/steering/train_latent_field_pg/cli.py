import os
import time
import argparse
import torch
from sae.utils import esm_utils
from sae.steering.pg.data_utils import load_data, split_dataset, GeneralDataset
from sae.steering.pg.surrogate_utils import load_or_train_surrogate
from sae.steering.pg.policy import StochasticLatentPolicy
from sae.steering.train_latent_field_pg.run_train import run_training
from sae.steering.train_latent_field_pg.cache_helpers import make_caches

def build_parser():
    p = argparse.ArgumentParser(description="Latent-field hybrid trainer (ΔZ + ŷ)")
    # Required
    p.add_argument("--ridge", required=True, help="Path to ridge surrogate (.joblib)")
    p.add_argument("--seqs", required=True, help="CSV with sequences and labels")

    # Optional
    p.add_argument("--scores", required=False)
    p.add_argument("--encoding", default="onehot", help="Encoding scheme for surrogate")
    p.add_argument("--esm", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="runs/poc_latent_field_pg/")
    p.add_argument("--sae", required=False)
    p.add_argument("--interplm_layer", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-epochs", type=int, default=100)
    p.add_argument("--m-range", nargs=3, type=float, default=[-1.0, 1.0, 21])
    p.add_argument("--m-bins", type=int, default=21, help="Number of discrete magnitude bins")
    p.add_argument("--policy-std", type=float, default=0.1)
    p.add_argument("--entropy-coef", type=float, default=1e-4)
    p.add_argument("--sparse-coef", type=float, default=1e-3)
    p.add_argument("--baseline-beta", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--use-mean-for-reward", action="store_true")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--cache-dir", default="cache/")
    p.add_argument("--persist-caches", action="store_true")
    p.add_argument("--cache-activations", action="store_true")
    p.add_argument("--cache-encodings", action="store_true")
    p.add_argument("--cache-decoded", action="store_true")
    return p

def main():
    start = time.time()
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.persist_caches:
        os.makedirs(args.cache_dir, exist_ok=True)

    seqs, labels, splits = load_data(args.seqs, args.scores)
    (train_x, train_y), (test_x, test_y) = split_dataset(seqs, labels, splits)
    esm_model, tokenizer = esm_utils.load_esm2_model(args.esm, device=args.device)
    if args.sae:
        sae_model = torch.load(args.sae, map_location=args.device)
    else:
        sae_model = esm_utils.load_interplm(args.esm, plm_layer=args.interplm_layer, device=args.device)

    ridge = load_or_train_surrogate(args.ridge, train_x, train_y, args.encoding)
    latent_dim = sae_model.decoder.out_features
    policy = StochasticLatentPolicy(latent_dim=latent_dim, sigma=args.policy_std).to(args.device)
    act_cache, enc_cache, dec_cache = make_caches(args)

    run_training(args, train_x, train_y, test_x, test_y,
                 sae_model, esm_model, tokenizer,
                 ridge, policy, act_cache, enc_cache, dec_cache)

    elapsed = time.time() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"⏱️ Total runtime: {int(h)}h {int(m)}m {s:.1f}s")

if __name__ == "__main__":
    main()

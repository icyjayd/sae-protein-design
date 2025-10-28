import argparse, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from sae.utils import esm_utils
from sae.steering.pg.data_utils import GeneralDataset, load_data, split_dataset
from sae.steering.pg.surrogate_utils import load_or_train_surrogate, predict_sequences
from sae.steering.pg.policy import StochasticLatentPolicy
from sae.steering.pg.perturb_utils import decode_with_deltas, get_activation_matrix
from sae.steering.pg.eval_utils import log_eval
from sae.steering.pg.cache_utils import ActivationCache, EncodingCache, DecodeCache

def train_policy_fast(policy, dataloader, sae_model, esm_model, tokenizer, ridge,
                      device, encoding, m_min, m_max, threshold,
                      entropy_coef, sparse_coef, baseline_beta, epochs, print_every,
                      train_x, train_y, test_x, test_y, eval_every,
                      use_mean_for_reward, m_bins, cache_activations, cache_decoded, cache_encodings):
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    baseline = 0.0

    act_cache = ActivationCache() if cache_activations else None
    dec_cache = DecodeCache() if cache_decoded else None
    enc_cache = EncodingCache() if cache_encodings else None

    def get_Z(seq):
        if act_cache is not None:
            z = act_cache.get(seq)
            if z is not None:
                return z
        z = get_activation_matrix(seq, sae_model, esm_model, tokenizer, device)
        if act_cache is not None:
            act_cache.set(seq, z)
        return z

    def quantize_m(m_val, m_min, m_max, bins):
        edges = np.linspace(m_min, m_max, int(bins))
        idx = np.argmin(np.abs(edges - m_val))
        return float(edges[idx])

    for epoch in range(epochs):
        total_loss = 0.0
        for seq, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
            seq = seq[0]
            m_val = np.random.uniform(m_min, m_max)
            m = torch.tensor(m_val, dtype=torch.float32, device=device)
            m_bin = quantize_m(m_val, m_min, m_max, m_bins)

            Z = get_Z(seq)
            dZ_samp, dZ_mean, logp, ent = policy.sample(Z, m)

            # Use mean for reward (deterministic) to improve decode-cache reuse
            mode = "mean" if use_mean_for_reward else "sample"
            dZ_for_reward = dZ_mean if use_mean_for_reward else dZ_samp

            seq_pert = None
            if dec_cache is not None:
                seq_pert = dec_cache.get(seq, m_bin, mode)
            if seq_pert is None:
                seq_pert = decode_with_deltas(seq, dZ_for_reward, sae_model, esm_model, tokenizer, device, threshold)
                if dec_cache is not None:
                    dec_cache.set(seq, m_bin, mode, seq_pert)

            y0 = predict_sequences(ridge, [seq], encoding, encoding_cache=enc_cache)[0]
            yp = predict_sequences(ridge, [seq_pert], encoding, encoding_cache=enc_cache)[0]

            reward = -((yp - (y0 + m_bin)) ** 2)  # use binned m for stable target
            baseline = (1 - baseline_beta) * baseline + baseline_beta * reward
            adv = reward - baseline

            sparse_pen = sparse_coef * dZ_samp.abs().mean()
            loss = -adv * logp + sparse_pen - (entropy_coef * ent)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        if epoch % print_every == 0:
            print(f"epoch {epoch} avg_obj {total_loss/len(dataloader):.4f} baseline {baseline:.4f}")
        if epoch % eval_every == 0:
            log_eval(epoch, ridge, encoding, train_x, train_y, test_x, test_y, encoding_cache=enc_cache)
    return policy

def sweep_codirectionality(seq, policy, sae_model, esm_model, tokenizer,
                           ridge, device, encoding, m_start, m_end, m_steps, threshold,
                           cache_decoded=False):
    results = []
    edges = np.linspace(m_start, m_end, int(m_steps))
    y0 = predict_sequences(ridge, [seq], encoding)[0]
    Z = get_activation_matrix(seq, sae_model, esm_model, tokenizer, device)
    dec_cache = DecodeCache() if cache_decoded else None
    for m_val in edges:
        m = torch.tensor(m_val, dtype=torch.float32, device=device)
        with torch.no_grad():
            dZ_mean = policy(Z, m)
        seq_pert = None
        if dec_cache is not None:
            seq_pert = dec_cache.get(seq, m_val, "mean")
        if seq_pert is None:
            seq_pert = decode_with_deltas(seq, dZ_mean, sae_model, esm_model, tokenizer, device, threshold)
            if dec_cache is not None:
                dec_cache.set(seq, m_val, "mean", seq_pert)
        yp = predict_sequences(ridge, [seq_pert], encoding)[0]
        results.append((float(m_val), float(yp - y0)))
    return results

def main():
    parser = argparse.ArgumentParser(description="Fast training with caches for activations, encodings, and decodes.")
    parser.add_argument("--seqs", required=True)
    parser.add_argument("--scores", required=False)
    parser.add_argument("--sae", required=False)
    parser.add_argument("--ridge", required=True)
    parser.add_argument("--encoding", type=str, default="aac")
    parser.add_argument("--esm", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="runs/poc_latent_field_pg_fast/")
    parser.add_argument("--m-range", nargs=3, type=float, default=[-1.0, 1.0, 21])
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--policy-std", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=1e-4)
    parser.add_argument("--sparse-coef", type=float, default=1e-3)
    parser.add_argument("--baseline-beta", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--interplm_layer", type=int, default=6)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--cache-activations", action="store_true")
    parser.add_argument("--cache-encodings", action="store_true")
    parser.add_argument("--cache-decoded", action="store_true")
    parser.add_argument("--use-mean-for-reward", action="store_true")
    parser.add_argument("--m-bins", type=int, default=21)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seqs, labels, splits = load_data(args.seqs, args.scores)
    (train_x, train_y), (test_x, test_y) = split_dataset(seqs, labels, splits)
    dataloader = DataLoader(GeneralDataset(train_x, train_y), batch_size=1, shuffle=True)

    esm_model, tokenizer = esm_utils.load_esm2_model(args.esm, device=args.device)
    if args.sae:
        print(f"[INFO] Loading custom SAE from {args.sae}")
        sae_model = torch.load(args.sae, map_location=args.device)
    else:
        print(f"[INFO] Loading InterPLM SAE corresponding to {args.esm}, layer {args.interplm_layer}")
        sae_model = esm_utils.load_interplm(args.esm, plm_layer=args.interplm_layer, device=args.device)

    ridge = load_or_train_surrogate(args.ridge, train_x, train_y, args.encoding)

    latent_dim = sae_model.decoder.out_features
    policy = StochasticLatentPolicy(latent_dim=latent_dim, sigma=args.policy_std).to(args.device)

    policy = train_policy_fast(policy, dataloader, sae_model, esm_model, tokenizer, ridge,
                               args.device, args.encoding, args.m_range[0], args.m_range[1], args.threshold,
                               args.entropy_coef, args.sparse_coef, args.baseline_beta, args.train_epochs, args.print_every,
                               train_x, train_y, test_x, test_y, args.eval_every,
                               args.use_mean_for_reward, args.m_bins, args.cache_activations, args.cache_decoded, args.cache_encodings)

    results = sweep_codirectionality(test_x[0], policy, sae_model, esm_model, tokenizer,
                                     ridge, args.device, args.encoding,
                                     args.m_range[0], args.m_range[1], args.m_range[2],
                                     args.threshold, cache_decoded=args.cache_decoded)

    with open(os.path.join(args.outdir, "codirectionality_pg.json"), "w") as f:
        json.dump(results, f, indent=2)

    mags, deltas = zip(*results)
    plt.figure(figsize=(5,4))
    plt.plot(mags, deltas, marker="o")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Requested magnitude (m)")
    plt.ylabel("Predicted property change (dy)")
    plt.title("Codirectionality Sweep (Policy Mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "codirectionality_pg.png"))
    plt.close()

    torch.save(policy.state_dict(), os.path.join(args.outdir, "latent_policy_pg_fast.pth"))
    print(f"✅ Training (fast) complete. Results saved in {args.outdir}")

if __name__ == "__main__":
    main()

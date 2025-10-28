import argparse, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from sae.utils import esm_utils
from sae.utils.esm_utils import build_perturbations_from_batch, perturb_and_decode_batch
from sae.steering.pg.data_utils import GeneralDataset, load_data, split_dataset
from sae.steering.pg.surrogate_utils import load_or_train_surrogate, predict_sequences
from sae.steering.pg.policy import StochasticLatentPolicy
from sae.steering.pg.eval_utils import log_eval
from sae.steering.pg.cache_utils import ActivationCache, EncodingCache, DecodeCache


# --------------------------- cache helpers ---------------------------

def make_caches(args):
    act = ActivationCache(args.cache_dir, args.persist_caches) if args.cache_activations else None
    enc = EncodingCache(args.cache_dir, args.persist_caches) if args.cache_encodings else None
    dec = DecodeCache(args.cache_dir, args.persist_caches) if args.cache_decoded else None
    return act, enc, dec


def cache_coverage(cache, seqs):
    """Return (n_cached, fraction) for how many seqs are in cache."""
    if cache is None or not hasattr(cache, "_cache"):
        return 0, 0.0
    n_cached = sum(1 for s in seqs if s in cache._cache)
    frac = n_cached / max(1, len(seqs))
    return n_cached, frac


# --------------------------- training utilities ---------------------------

def quantize_m(m_vals, m_min, m_max, bins):
    edges = np.linspace(m_min, m_max, int(bins))
    idxs = np.abs(edges[None, :] - np.array(m_vals)[:, None]).argmin(-1)
    return edges[idxs].astype(float)


def get_Z_cached(seq, sae_model, esm_model, tokenizer, device, act_cache):
    if act_cache is not None:
        z = act_cache.get(seq)
        if z is not None:
            return z
    z = esm_utils.get_activation_matrix(seq, sae_model, esm_model, tokenizer, device=device)
    if act_cache is not None:
        act_cache.set(seq, z.cpu())
    return z


def decode_batched_with_cache(seq_batch, dZ_batch, sae_model, esm_model, tokenizer,
                              device, threshold, m_bins_for_key, dec_cache, use_mean):
    """Decode a batch, using cache hits when possible."""
    mode = "mean" if use_mean else "sample"
    decoded = [None] * len(seq_batch)
    need_idx, need_seqs, need_dZ = [], [], []

    # Try cache
    for i, (s, mbin) in enumerate(zip(seq_batch, m_bins_for_key)):
        cached = dec_cache.get(s, mbin, mode) if dec_cache is not None else None
        if cached is None:
            need_idx.append(i); need_seqs.append(s); need_dZ.append(dZ_batch[i])
        else:
            decoded[i] = cached

    # Batch decode only missing
    if need_idx:
        pert = esm_utils.build_perturbations_from_batch(need_dZ, threshold=threshold)
        dec_out = esm_utils.perturb_and_decode_batch(need_seqs, sae_model, esm_model, tokenizer,
                                                     pert, device=device, threshold=threshold, dtype=torch.float16)
        for k, seq_pert in zip(need_idx, dec_out):
            decoded[k] = seq_pert
            if dec_cache is not None:
                dec_cache.set(seq_batch[k], float(m_bins_for_key[k]), mode, seq_pert)
    return decoded


# --------------------------- training loop ---------------------------

def train_epoch_batched(policy, batch, sae_model, esm_model, tokenizer, ridge, device, encoding,
                        m_lo, m_hi, threshold, entropy_coef, sparse_coef, baseline, baseline_beta,
                        act_cache, enc_cache, dec_cache, use_mean_for_reward, m_bins):
    """One epoch step over a DataLoader batch."""
    seq_batch, _ = zip(*batch)
    B = len(seq_batch)
    m_vals = np.random.uniform(m_lo, m_hi, size=B).astype(np.float32)
    m_bins_for_key = quantize_m(m_vals, m_lo, m_hi, m_bins)
    m_torch = torch.tensor(m_vals, dtype=torch.float32, device=device)

    # activations (cached)
    Z_list = [get_Z_cached(s, sae_model, esm_model, tokenizer, device, act_cache) for s in seq_batch]

    # policy sampling
    dZ_samp_list, dZ_mean_list, logp_list, ent_list = [], [], [], []
    for Z_i, m_i in zip(Z_list, m_torch):
        dZ_samp, dZ_mean, logp, ent = policy.sample(Z_i, m_i)
        dZ_samp_list.append(dZ_samp); dZ_mean_list.append(dZ_mean)
        logp_list.append(logp); ent_list.append(ent)

    dZ_for_reward = dZ_mean_list if use_mean_for_reward else dZ_samp_list
    seq_pert_batch = decode_batched_with_cache(
        list(seq_batch), dZ_for_reward, sae_model, esm_model, tokenizer,
        device, threshold, m_bins_for_key, dec_cache, use_mean_for_reward
    )

    # Ridge predictions
    y0 = predict_sequences(ridge, list(seq_batch), encoding, encoding_cache=enc_cache)
    yp = predict_sequences(ridge, list(seq_pert_batch), encoding, encoding_cache=enc_cache)

    rewards = -((yp - (y0 + m_bins_for_key)) ** 2)
    baseline = (1 - baseline_beta) * baseline + baseline_beta * rewards.mean()
    advantages = rewards - baseline

    sparse_term = torch.stack([d.abs().mean() for d in dZ_samp_list]).mean() * sparse_coef
    ent_term = torch.stack(ent_list).mean() * entropy_coef
    logp_sum = torch.stack(logp_list).mean()
    loss = -torch.tensor(advantages.mean(), dtype=torch.float32, device=device) * logp_sum + sparse_term - ent_term
    return loss, float(rewards.mean()), float(baseline)


def train_loop(policy, dataloader, sae_model, esm_model, tokenizer, ridge,
               device, encoding, m_range, threshold, entropy_coef, sparse_coef, baseline_beta,
               epochs, print_every, eval_every, act_cache, enc_cache, dec_cache,
               use_mean_for_reward, m_bins, train_x, train_y, test_x, test_y, persist_caches=False):
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    baseline = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss, mean_reward, baseline = train_epoch_batched(
                policy, list(zip(batch[0], batch[1])), sae_model, esm_model, tokenizer, ridge, device, encoding,
                m_range[0], m_range[1], threshold, entropy_coef, sparse_coef, baseline, baseline_beta,
                act_cache, enc_cache, dec_cache, use_mean_for_reward, m_bins
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        print(f"epoch {epoch} avg_obj {total_loss/max(1,len(dataloader)):.4f} baseline {baseline:.4f}")

        if epoch % eval_every == 0:
            log_eval(epoch, ridge, encoding, train_x, train_y, test_x, test_y, encoding_cache=enc_cache)

        # ---------------- cache diagnostics + save per epoch ----------------
        if persist_caches:
            for name, c, seqs in [
                ("activations", act_cache, train_x),
                ("encodings", enc_cache, train_x),
                ("decodes", dec_cache, train_x),
            ]:
                if c is not None:
                    n, frac = cache_coverage(c, seqs)
                    print(f"[cache] {name}: {n}/{len(seqs)} cached ({frac*100:.1f}%)")
                    c.save()
            print(f"[cache] Saved all caches after epoch {epoch}")

    return policy


# --------------------------- main ---------------------------

def main():
    p = argparse.ArgumentParser(description="Batched REINFORCE latent-field trainer with per-epoch cache saves")
    p.add_argument("--seqs", required=True)
    p.add_argument("--scores", required=False)
    p.add_argument("--sae", required=False)
    p.add_argument("--ridge", required=True)
    p.add_argument("--encoding", default="aac")
    p.add_argument("--esm", default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="runs/poc_latent_field_pg_fast/")
    p.add_argument("--m-range", nargs=3, type=float, default=[-1.0, 1.0, 21])
    p.add_argument("--train-epochs", type=int, default=100)
    p.add_argument("--policy-std", type=float, default=0.1)
    p.add_argument("--entropy-coef", type=float, default=1e-4)
    p.add_argument("--sparse-coef", type=float, default=1e-3)
    p.add_argument("--baseline-beta", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--interplm_layer", type=int, default=6)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    # caching
    p.add_argument("--cache-dir", default="cache/")
    p.add_argument("--persist-caches", action="store_true")
    p.add_argument("--cache-activations", action="store_true")
    p.add_argument("--cache-encodings", action="store_true")
    p.add_argument("--cache-decoded", action="store_true")
    # decoding
    p.add_argument("--use-mean-for-reward", action="store_true")
    p.add_argument("--m-bins", type=int, default=21)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.persist_caches:
        os.makedirs(args.cache_dir, exist_ok=True)

    seqs, labels, splits = load_data(args.seqs, args.scores)
    (train_x, train_y), (test_x, test_y) = split_dataset(seqs, labels, splits)
    dl = DataLoader(GeneralDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=False)

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

    act_cache, enc_cache, dec_cache = make_caches(args)

    policy = train_loop(
        policy, dl, sae_model, esm_model, tokenizer, ridge,
        args.device, args.encoding, args.m_range, args.threshold,
        args.entropy_coef, args.sparse_coef, args.baseline_beta,
        args.train_epochs, args.print_every, args.eval_every,
        act_cache, enc_cache, dec_cache,
        args.use_mean_for_reward, args.m_bins,
        train_x, train_y, test_x, test_y,
        persist_caches=args.persist_caches
    )

    # Save final model
    torch.save(policy.state_dict(), os.path.join(args.outdir, "latent_policy_pg_fast.pth"))
    print(f"✅ Done. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()

import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from sae.utils import esm_utils
from ml_models.encoding import encode_sequences
from sklearn.linear_model import Ridge
import joblib


# ================================================================
#   Dataset
# ================================================================

class GeneralDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labels[i]


def load_data(seq_path, label_path=None):
    import pandas as pd
    if seq_path.endswith(".csv"):
        df = pd.read_csv(seq_path)
        # ✅ Changed "score" -> "label"
        if label_path is None and "label" in df.columns:
            seqs = df["sequence"].tolist()
            scores = df["label"].to_numpy()
        elif label_path is not None:
            seqs = df["sequence"].tolist()
            scores = np.load(label_path)
        else:
            raise ValueError("Need either 'label' column or external label file.")
        splits = df["split"].tolist() if "split" in df.columns else None
    elif seq_path.endswith(".npy") and label_path is not None:
        seqs = np.load(seq_path, allow_pickle=True)
        scores = np.load(label_path)
        splits = None
    else:
        raise ValueError("Unsupported file combination.")
    return seqs, scores, splits


def split_dataset(seqs, scores, splits):
    if splits is None:
        n = len(seqs); idx = int(0.9 * n)
        return (seqs[:idx], scores[:idx]), (seqs[idx:], scores[idx:])
    train_idx = [i for i,s in enumerate(splits) if str(s).lower()=="train"]
    test_idx  = [i for i,s in enumerate(splits) if str(s).lower()=="test"]
    seqs, scores = np.array(seqs), np.array(scores)
    return (seqs[train_idx], scores[train_idx]), (seqs[test_idx], scores[test_idx])


# ================================================================
#   Stochastic latent policy
# ================================================================

class StochasticLatentPolicy(nn.Module):
    """Outputs Normal(mu,sigma) over latent deltas per position/feature."""
    def __init__(self, latent_dim: int, hidden: int = 128, sigma: float = 0.1):
        super().__init__()
        self.m_to_emb = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim)
        )
        self.scale = nn.Parameter(torch.ones(latent_dim))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
    def forward(self, Z: torch.Tensor, m: torch.Tensor):
        base = self.m_to_emb(m.view(1,1))
        dZ_mean = (base.squeeze(0) * self.scale * 0.1).expand(Z.shape[0], -1)
        return dZ_mean
    def sample(self, Z: torch.Tensor, m: torch.Tensor):
        dZ_mean = self.forward(Z, m)
        dist = Normal(dZ_mean, self.sigma)
        dZ = dist.rsample()
        logp = dist.log_prob(dZ).sum()
        ent = dist.entropy().sum()
        return dZ, dZ_mean, logp, ent


# ================================================================
#   Perturbation and decoding helpers
# ================================================================

def decode_with_deltas(sequence, dZ, sae_model, esm_model, tokenizer, device, threshold):
    L, N = dZ.shape
    perturb = {}
    dZ_cpu = dZ.detach().cpu().numpy()
    for i in range(L):
        edits = {}
        for j in range(N):
            val = float(dZ_cpu[i, j])
            if abs(val) > threshold:
                edits[j] = val
        if edits:
            perturb[i+1] = edits
    seq_pert = esm_utils.perturb_and_decode(
        sequence=sequence,
        sae_model=sae_model,
        esm_model=esm_model,
        tokenizer=tokenizer,
        surgical_perturbations=perturb,
        device=device,
    )
    return seq_pert, perturb


# ================================================================
#   Training with REINFORCE
# ================================================================

def train_pg(policy, dataloader, sae_model, esm_model, tokenizer, ridge,
             device, encoding, m_min, m_max, threshold,
             entropy_coef, sparse_coef, baseline_beta, epochs):
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    baseline = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        for seq, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
            seq = seq[0]
            m_val = np.random.uniform(m_min, m_max)
            m = torch.tensor(m_val, dtype=torch.float32, device=device)

            Z = esm_utils.get_activation_matrix(seq, sae_model, esm_model, tokenizer, device=device)
            dZ, dZ_mean, logp, ent = policy.sample(Z, m)

            seq_pert, _ = decode_with_deltas(seq, dZ, sae_model, esm_model, tokenizer, device, threshold)

            # Ridge surrogate predictions use same encoder for both
            X0 = encode_sequences([seq], encoding=encoding)
            Xp = encode_sequences([seq_pert], encoding=encoding)
            y0 = ridge.predict(X0)[0]
            yp = ridge.predict(Xp)[0]

            err = yp - (y0 + m_val)
            reward = - (err ** 2)
            baseline = (1 - baseline_beta) * baseline + baseline_beta * reward
            adv = reward - baseline

            sparse_pen = sparse_coef * dZ.abs().mean()
            entropy_bonus = entropy_coef * ent
            loss = -adv * logp + sparse_pen - entropy_bonus

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        if epoch % 10 == 0:
            print(f"epoch {epoch} avg_obj {total_loss/len(dataloader):.4f} baseline {baseline:.4f}")
    return policy


# ================================================================
#   Evaluation (codirectionality sweep)
# ================================================================

def sweep_codirectionality(seq, policy, sae_model, esm_model, tokenizer,
                           ridge, device, encoding, m_start, m_end, m_steps, threshold):
    mags = np.linspace(m_start, m_end, int(m_steps))
    results = []
    y0 = ridge.predict(encode_sequences([seq], encoding=encoding))[0]
    Z = esm_utils.get_activation_matrix(seq, sae_model, esm_model, tokenizer, device=device)
    for m_val in mags:
        m = torch.tensor(m_val, dtype=torch.float32, device=device)
        with torch.no_grad():
            dZ_mean = policy(Z, m)
        seq_pert, _ = decode_with_deltas(seq, dZ_mean, sae_model, esm_model, tokenizer, device, threshold)
        yp = ridge.predict(encode_sequences([seq_pert], encoding=encoding))[0]
        results.append((float(m_val), float(yp - y0)))
    return results


# ================================================================
#   CLI Entry Point
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train stochastic latent policy with REINFORCE using ridge on reconstructed sequences.")
    parser.add_argument("--seqs", required=True, help="CSV or NPY with sequences")
    parser.add_argument("--scores", required=False, help="NPY or CSV with labels")
    parser.add_argument("--sae", required=False, help="Path to trained SAE (.pt). If omitted, loads InterPLM SAE")
    parser.add_argument("--ridge", required=True, help="Path to ridge surrogate pickle")
    parser.add_argument("--encoding", type=str, default="aac", help="Encoding type for surrogate (aac, dpc, esm, etc.)")
    parser.add_argument("--esm", default="facebook/esm2_t6_8M_UR50D", help="ESM model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="runs/poc_latent_field_pg/")
    parser.add_argument("--m-range", nargs=3, type=float, default=[-1.0, 1.0, 11])
    parser.add_argument("--train-epochs", type=int, default=150)
    parser.add_argument("--policy-std", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=1e-4)
    parser.add_argument("--sparse-coef", type=float, default=1e-3)
    parser.add_argument("--baseline-beta", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--interplm_layer", type=int, default=6, help="Layer for InterPLM SAE (default 6)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seqs, scores, splits = load_data(args.seqs, args.scores)
    (train_x, train_y), (test_x, test_y) = split_dataset(seqs, scores, splits)
    dataloader = DataLoader(GeneralDataset(train_x, train_y), batch_size=1, shuffle=True)

    esm_model, tokenizer = esm_utils.load_esm2_model(args.esm, device=args.device)
    if args.sae:
        print(f"[INFO] Loading custom SAE from {args.sae}")
        sae_model = torch.load(args.sae, map_location=args.device)
    else:
        print(f"[INFO] Loading InterPLM SAE corresponding to {args.esm}, layer {args.interplm_layer}")
        sae_model = esm_utils.load_interplm(args.esm, plm_layer=args.interplm_layer, device=args.device)

    # build or load surrogate
    if os.path.exists(args.ridge):
        print(f"[INFO] Loading pretrained surrogate from {args.ridge}")
        ridge = joblib.load(args.ridge)
    else:
        print(f"[INFO] Training new Ridge surrogate using encoding '{args.encoding}'")
        X = encode_sequences(train_x, encoding=args.encoding)
        y = np.array(train_y)
        ridge = Ridge(alpha=1.0).fit(X, y)
        joblib.dump(ridge, args.ridge)
        print(f"[INFO] Saved surrogate to {args.ridge}")

    latent_dim = sae_model.decoder.out_features
    policy = StochasticLatentPolicy(latent_dim=latent_dim, sigma=args.policy_std).to(args.device)

    policy = train_pg(policy, dataloader, sae_model, esm_model, tokenizer, ridge,
                      args.device, args.encoding, args.m_range[0], args.m_range[1],
                      args.threshold, args.entropy_coef, args.sparse_coef,
                      args.baseline_beta, args.train_epochs)

    results = sweep_codirectionality(test_x[0], policy, sae_model, esm_model, tokenizer,
                                     ridge, args.device, args.encoding,
                                     args.m_range[0], args.m_range[1], args.m_range[2],
                                     args.threshold)

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

    torch.save(policy.state_dict(), os.path.join(args.outdir, "latent_policy_pg.pth"))
    print(f"✅ Training complete. Results saved in {args.outdir}")


if __name__ == "__main__":
    main()

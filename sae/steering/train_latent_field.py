import argparse, os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from sae.utils import esm_utils
from sae.steering.poc_latent_field import LatentFieldPOC


# ================================================================
#   Dataset Handling
# ================================================================

class GeneralDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def load_data(seq_path, label_path=None):
    import pandas as pd
    if seq_path.endswith(".csv"):
        df = pd.read_csv(seq_path)
        if label_path is None and "score" in df.columns:
            seqs = df["sequence"].tolist()
            scores = df["score"].to_numpy()
        elif label_path is not None:
            seqs = df["sequence"].tolist()
            scores = np.load(label_path)
        else:
            raise ValueError("Need either 'score' column or external score file.")
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
        n = len(seqs)
        idx = int(0.9 * n)
        return (seqs[:idx], scores[:idx]), (seqs[idx:], scores[idx:])
    else:
        train_idx = [i for i, s in enumerate(splits) if str(s).lower() == "train"]
        test_idx = [i for i, s in enumerate(splits) if str(s).lower() == "test"]
        seqs, scores = np.array(seqs), np.array(scores)
        return (seqs[train_idx], scores[train_idx]), (seqs[test_idx], scores[test_idx])


# ================================================================
#   Core Utilities
# ================================================================

def surrogate_eval(ridge_surrogate, Z):
    with torch.no_grad():
        if hasattr(ridge_surrogate, "predict"):
            y = ridge_surrogate.predict(Z.cpu().numpy().reshape(1, -1))
            return torch.tensor(y[0], dtype=torch.float32)
        else:
            return ridge_surrogate(Z.unsqueeze(0)).squeeze()


def forward_with_delta(sequence, sae_model, esm_model, tokenizer, field, m, device="cpu"):
    Z = esm_utils.get_activation_matrix(sequence, sae_model, esm_model, tokenizer, device=device)
    dZ = field(Z.to(device), m.to(device))
    Zp = Z.to(device) + dZ
    return Z.to(device), Zp


def sweep_codirectionality(seq, field, sae_model, esm_model, tokenizer,
                           ridge_surrogate, device, m_start, m_end, m_steps):
    mags = torch.linspace(m_start, m_end, m_steps)
    results = []
    for m in mags:
        Z, Zp = forward_with_delta(seq, sae_model, esm_model, tokenizer, field, m, device=device)
        y0 = surrogate_eval(ridge_surrogate, Z.mean(0))
        yp = surrogate_eval(ridge_surrogate, Zp.mean(0))
        results.append((float(m.item()), float((yp - y0).item())))
    return results


def train_latent_field(field, dataloader, sae_model, esm_model, tokenizer,
                       ridge_surrogate, device):
    opt = torch.optim.Adam(field.parameters(), lr=1e-3)
    for epoch in range(300):
        total = 0
        for seq, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
            seq = seq[0]
            m = torch.tensor(0.5, device=device)
            Z, Zp = forward_with_delta(seq, sae_model, esm_model, tokenizer, field, m, device)
            y0 = surrogate_eval(ridge_surrogate, Z.mean(0))
            yp = surrogate_eval(ridge_surrogate, Zp.mean(0))
            loss_mag = (yp - (y0 + m)) ** 2
            loss_sparse = 1e-3 * Zp.sub(Z).abs().mean()
            loss = loss_mag + loss_sparse
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: mean loss {total/len(dataloader):.4f}")
    return field


# ================================================================
#   CLI Entry Point
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LatentFieldPOC on sequence+score data.")
    parser.add_argument("--seqs", required=True, help="CSV or NPY with sequences")
    parser.add_argument("--scores", required=False, help="NPY or CSV with scores")
    parser.add_argument("--sae", required=False, help="Path to trained SAE (.pt). If not given, loads InterPLM SAE.")
    parser.add_argument("--ridge", required=True, help="Path to ridge surrogate (pickle or torch)")
    parser.add_argument("--esm", default="facebook/esm2_t6_8M_UR50D", help="ESM model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="runs/poc_latent_field/")
    parser.add_argument("--m-range", nargs=3, type=float, default=[-1.0, 1.0, 11],
                        metavar=("START", "END", "STEPS"),
                        help="Range of magnitudes for codirectionality sweep (default: -1 1 11)")
    parser.add_argument("--interplm-model", default="interplm/interplm-sae", help="InterPLM SAE model name on HF")
    parser.add_argument("--interplm-layer", type=int, default=20, help="Layer index for SAE (default 20)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Data ---
    seqs, scores, splits = load_data(args.seqs, args.scores)
    (train_x, train_y), (test_x, test_y) = split_dataset(seqs, scores, splits)
    dataset = GeneralDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --- Models ---
    esm_model, tokenizer = esm_utils.load_esm2_model(args.esm, device=args.device)

    if args.sae:
        print(f"Loading custom SAE from {args.sae}")
        sae_model = torch.load(args.sae, map_location=args.device)
    else:
        print(f"Loading default InterPLM SAE from {args.interplm_model}, layer {args.interplm_layer}")
        from interplm.sae.inference import load_sae_from_hf
        sae_model = load_sae_from_hf(args.interplm_model, plm_layer=args.interplm_layer)
        sae_model.to(args.device)
        sae_model.eval()

    import joblib
    ridge_surrogate = joblib.load(args.ridge)

    latent_dim = sae_model.decoder.out_features
    field = LatentFieldPOC(latent_dim=latent_dim).to(args.device)

    # --- Train ---
    trained_field = train_latent_field(
        field, dataloader, sae_model, esm_model, tokenizer, ridge_surrogate, args.device
    )

    # --- Codirectionality Evaluation ---
    m_start, m_end, m_steps = args.m_range
    test_seq = test_x[0]
    results = sweep_codirectionality(
        test_seq, trained_field, sae_model, esm_model, tokenizer,
        ridge_surrogate, args.device,
        m_start, m_end, int(m_steps)
    )

    json_path = os.path.join(args.outdir, "codirectionality.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    mags, deltas = zip(*results)
    plt.figure(figsize=(5,4))
    plt.plot(mags, deltas, marker="o")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Requested magnitude (m)")
    plt.ylabel("Predicted change in property (dy)")
    plt.title("Codirectionality Sweep")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "codirectionality.png"))
    plt.close()

    torch.save(trained_field.state_dict(), os.path.join(args.outdir, "latent_field.pth"))
    print(f"✅ Training complete. Results saved in {args.outdir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, numpy as np, torch
from pathlib import Path
from utils.model_utils import SparseAutoencoderSAE, MonosemanticSAE
import matplotlib.pyplot as plt

OUT = Path("outputs")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["regular", "monosemantic", "both"], default="both")
    ap.add_argument("--threshold-pct", type=float, default=70)
    args = ap.parse_args()

    acts = np.load(OUT / "activations.npy")
    acts = torch.tensor(acts, dtype=torch.float32)

    if args.mode in ("regular", "both"):
        model = SparseAutoencoderSAE(acts.shape[1], 64, 512)
        model.load_state_dict(torch.load(OUT / "sae_regular.pt", map_location="cpu"), strict=False)
        model.eval()
        codes = model.encode(acts).detach().numpy()
        np.save(OUT / "sparse_codes_regular.npy", codes)
        np.save(OUT / "sae_atoms_regular.npy", model.decoder.weight.detach().numpy())
        print("Saved sparse_codes_regular.npy and sae_atoms_regular.npy")

    if args.mode in ("monosemantic", "both"):
        model = MonosemanticSAE(acts.shape[1], 64, 512)

        # --- Flexible state_dict loader for pretrained InterPLM SAEs ---
        state_path = OUT / "sae_mono.pt"
        if not state_path.exists():
            raise FileNotFoundError("Expected pretrained SAE weights at outputs/sae_mono.pt")

        print("[INFO] Loading SAE weights (flexible mapping enabled)")
        state_dict = torch.load(state_path, map_location="cpu")

        mapped_state_dict = {}
        for k, v in state_dict.items():
            # Map single-layer InterPLM encoder/decoder to local Sequential structure
            if k.startswith("encoder.") and not k.startswith("encoder.0."):
                mapped_state_dict[k.replace("encoder.", "encoder.0.")] = v
            elif k.startswith("decoder.") and not k.startswith("decoder.0."):
                mapped_state_dict[k.replace("decoder.", "decoder.0.")] = v
            else:
                mapped_state_dict[k] = v

        # Non-strict loading: ignore irrelevant InterPLM-specific keys
        model.load_state_dict(mapped_state_dict, strict=False)
        print("[INFO] Pretrained SAE loaded successfully")

        model.eval()
        codes = model.encode(acts).detach().numpy()
        np.save(OUT / "sparse_codes_mono.npy", codes)
        np.save(OUT / "sae_atoms_mono.npy", model.decoder[0].weight.detach().numpy())
        print("Saved sparse_codes_mono.npy and sae_atoms_mono.npy")

    # Visual threshold diagnostic
    codes = np.load(OUT / f"sparse_codes_{'mono' if args.mode == 'monosemantic' else 'regular'}.npy")
    thresh = np.percentile(np.abs(codes), args.threshold_pct)
    plt.hist(np.abs(codes).ravel(), bins=100)
    plt.axvline(thresh, color='r', linestyle='--', label=f'{args.threshold_pct}th pct')
    plt.legend()
    plt.title("Activation Magnitude Distribution")
    plt.savefig(OUT / "activation_distribution.png")
    plt.close()

if __name__ == "__main__":
    main()

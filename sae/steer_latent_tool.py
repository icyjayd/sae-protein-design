# steer_latent.py

import argparse
import numpy as np
import torch
from utils.model_utils import MonosemanticSAE
from utils.esm_utils import load_esm2_model, decode_activation
from pathlib import Path
import json


OUTDIR = Path("outputs")


def load_codes_and_model(latent_index):
    codes = np.load(OUTDIR / "sparse_codes_mono.npy")
    atoms = np.load(OUTDIR / "sae_atoms_mono.npy")

    # Load SAE
    model = MonosemanticSAE(atoms.shape[1], 64, 512)
    model.decoder[0].weight.data = torch.tensor(atoms, dtype=torch.float32)
    model.eval()

    return codes, model


def steer_latent(codes, model, seq_idx, latent_idx, delta):
    orig = codes[seq_idx].copy()
    modified = orig.copy()
    modified[latent_idx] += delta

    # Decode to activation space
    orig_act = model.decode(torch.tensor(orig).unsqueeze(0)).detach().numpy()
    mod_act = model.decode(torch.tensor(modified).unsqueeze(0)).detach().numpy()

    return orig_act, mod_act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-idx", type=int, required=True, help="Index of base sequence")
    parser.add_argument("--latent", type=int, required=True, help="Latent to steer")
    parser.add_argument("--delta", type=float, required=True, help="Delta to apply to latent")
    parser.add_argument("--property", type=str, default="Property", help="Name of the property")
    args = parser.parse_args()

    codes, model = load_codes_and_model(args.latent)

    orig_act, mod_act = steer_latent(codes, model, args.seq_idx, args.latent, args.delta)

    # Load ESM2 and decode back to sequence
    esm_model, alphabet = load_esm2_model("esm2_t6_8M_UR50D")  # adjust as needed
    orig_seq = decode_activation(orig_act, esm_model, alphabet)
    mod_seq = decode_activation(mod_act, esm_model, alphabet)

    results = {
        "base_sequence_index": args.seq_idx,
        "latent_steered": args.latent,
        "delta": args.delta,
        "property": args.property,
        "original_sequence": orig_seq,
        "steered_sequence": mod_seq,
    }

    with open(OUTDIR / "steering_result.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[INFO] Steering complete. Result written to steering_result.json")


if __name__ == "__main__":
    main()
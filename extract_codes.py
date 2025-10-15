#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from utils.model_utils import SparseAutoencoderSAE, MonosemanticSAE

OUT = Path("outputs")

# --- A tiny wrapper compatible with InterPLM checkpoints ----------------------
class InterPLMCompat(torch.nn.Module):
    """
    Minimal encoder/decoder that matches InterPLM-style checkpoints:
      - keys: encoder.weight (latent x input), encoder.bias (latent),
              decoder.weight (input x latent), bias (input)
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = torch.nn.Linear(latent_dim, input_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z


def load_numpy(name: str):
    p = OUT / name
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Make sure you ran feature extraction first.")
    return np.load(p)


def save_atoms_via_decode(model, latent_dim: int, suffix: str):
    """Decode unit basis to get atoms consistently for any model backend."""
    with torch.no_grad():
        I = torch.eye(latent_dim, dtype=torch.float32)
        dec = model.decode(I).cpu().numpy()  # shape: (latent_dim, input_dim)
    np.save(OUT / f"sae_atoms_{suffix}.npy", dec)


def encode_and_save_codes(model, acts: np.ndarray, suffix: str, threshold_pct: float):
    x = torch.tensor(acts, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        z = model.encode(x).cpu().numpy()

    # Apply percentile sparsification (like your original pipeline)
    thresh = np.percentile(np.abs(z), threshold_pct)
    z_sparse = z * (np.abs(z) >= thresh)
    np.save(OUT / f"sparse_codes_{suffix}.npy", z_sparse)
    return z_sparse, thresh


def load_mono_model_flex(acts: np.ndarray):
    """
    Try to load the monosemantic SAE from outputs/sae_mono.pt.
    If it looks like InterPLM (flat keys), build InterPLMCompat.
    Otherwise, load into local MonosemanticSAE with flexible mapping.
    """
    state_path = OUT / "sae_mono.pt"
    if not state_path.exists():
        raise FileNotFoundError("Expected pretrained SAE weights at outputs/sae_mono.pt")

    state = torch.load(state_path, map_location="cpu")

    # Heuristic: InterPLM-style if it has 'encoder.weight' and 'decoder.weight'
    is_interplm = ("encoder.weight" in state) and ("decoder.weight" in state)

    if is_interplm:
        # Infer sizes from weight shapes
        enc_w = state["encoder.weight"]  # (latent, input)
        dec_w = state["decoder.weight"]  # (input, latent)
        latent_dim, input_dim = enc_w.shape[0], enc_w.shape[1]
        assert dec_w.shape == (input_dim, latent_dim), "decoder.weight shape mismatch with encoder.weight"

        model = InterPLMCompat(input_dim=input_dim, latent_dim=latent_dim)
        # Map keys into standard Linear layers
        mapped = {}
        mapped["encoder.weight"] = enc_w
        mapped["encoder.bias"] = state.get("encoder.bias", torch.zeros(latent_dim))
        mapped["decoder.weight"] = dec_w
        # InterPLM may store output bias under "bias"
        mapped["decoder.bias"] = state.get("bias", torch.zeros(input_dim))
        model.load_state_dict(mapped, strict=True)
        suffix = "mono"  # keep standard suffix
        return model, suffix, latent_dim

    # Otherwise: local MonosemanticSAE — allow flexible mapping from flat to Sequential
    input_dim = acts.shape[1]
    # If we have a saved config, try to read latent_dim; else infer from encoder weight
    latent_dim = None
    cfg_path = OUT / "sae_mono_config.json"
    if cfg_path.exists():
        import json
        cfg = json.loads(cfg_path.read_text())
        latent_dim = int(cfg["latent_dim"])
    if latent_dim is None:
        # Try to infer from available keys
        if "encoder.2.weight" in state:
            latent_dim = state["encoder.2.weight"].shape[0]
        elif "encoder.weight" in state:
            latent_dim = state["encoder.weight"].shape[0]
        else:
            raise RuntimeError("Cannot infer latent_dim for local MonosemanticSAE.")

    model = MonosemanticSAE(input_dim=input_dim, latent_dim=latent_dim, hidden=512, topk=None)

    # Flexible key remapping for local Sequential structure
    mapped_state = {}
    for k, v in state.items():
        if k.startswith("encoder.") and not k.startswith("encoder.0."):
            mapped_state[k.replace("encoder.", "encoder.0.")] = v
        elif k.startswith("decoder.") and not k.startswith("decoder.0."):
            mapped_state[k.replace("decoder.", "decoder.0.")] = v
        else:
            mapped_state[k] = v

    model.load_state_dict(mapped_state, strict=False)
    suffix = "mono"
    return model, suffix, latent_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["regular", "monosemantic", "both"], default="both")
    ap.add_argument("--threshold-pct", type=float, default=70)
    ap.add_argument("--outdir", nargs="?", default="outputs")
    args = ap.parse_args()
    global OUT
    OUT = Path(args.outdir)
    OUT.mkdir(exist_ok=True)

    acts = load_numpy("activations.npy")

    # REGULAR
    if args.mode in ("regular", "both"):
        # Load local regular SAE
        reg_path = OUT / "sae_regular.pt"
        if not reg_path.exists():
            print("[regular] No sae_regular.pt found; skipping regular path.")
        else:
            # Determine dims
            input_dim = acts.shape[1]
            # Try to read config for latent_dim
            latent_dim = 64
            cfg_path = OUT / "sae_regular_config.json"
            if cfg_path.exists():
                import json
                try:
                    cfg = json.loads(cfg_path.read_text())
                    latent_dim = int(cfg.get("latent_dim", latent_dim))
                except Exception:
                    pass

            reg = SparseAutoencoderSAE(input_dim=input_dim, latent_dim=latent_dim, hidden=512)
            reg.load_state_dict(torch.load(reg_path, map_location="cpu"), strict=False)
            codes_reg, thr_reg = encode_and_save_codes(reg, acts, "regular", args.threshold_pct)
            save_atoms_via_decode(reg, reg.latent_dim, "regular")
            print(f"[regular] Saved sparse_codes_regular.npy, sae_atoms_regular.npy (threshold={thr_reg:.4g})")

    # MONO
    if args.mode in ("monosemantic", "both"):
        try:
            model_mono, suffix, latent_dim = load_mono_model_flex(acts)
        except FileNotFoundError:
            print("[mono] No sae_mono.pt found; skipping monosemantic path.")
        else:
            codes_mono, thr_mono = encode_and_save_codes(model_mono, acts, suffix, args.threshold_pct)
            save_atoms_via_decode(model_mono, latent_dim, suffix)
            print(f"[{suffix}] Saved sparse_codes_{suffix}.npy, sae_atoms_{suffix}.npy (threshold={thr_mono:.4g})")


if __name__ == "__main__":
    main()

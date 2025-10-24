# sae/decoder/sample.py
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from .models import build_decoder
from .data import IDX_TO_AA


def decode_logits_to_sequence(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    indices = torch.multinomial(probs, 1).squeeze(-1)
    seq = "".join(IDX_TO_AA.get(i.item(), "X") for i in indices)
    return seq


def sample_sequences(model_path, latent_dim, model_type="gru",
                     n_samples=5, temperature=1.0, device="cpu", max_len=256, latents_path=None):
    """
    Load a trained decoder and generate new amino-acid sequences.
    Now infers MLP max_len automatically from checkpoint if needed.
    """
    checkpoint = torch.load(model_path, map_location=device)

    # FIX: infer max_len from checkpoint if possible
    inferred_max_len = max_len
    if model_type == "mlp":
        for k, v in checkpoint.items():
            if "model.4.weight" in k and v.ndim == 2:
                inferred_max_len = v.shape[0] // 20  # divide by vocab size
                print(f"[INFO] Inferred MLP max_len = {inferred_max_len}")
                break

    model = build_decoder(model_type, latent_dim, max_len=inferred_max_len)
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    sequences = []
    with torch.no_grad():
        if latents_path:
            latents = np.load(latents_path, allow_pickle=True)
            latents = torch.tensor(latents, dtype=torch.float32, device=device)
            if latents.ndim == 1:
                latents = latents.unsqueeze(0)
        else:
            latents = torch.randn(n_samples, latent_dim, device=device)

        for i, z in enumerate(latents):
            logits = model(z.unsqueeze(0), target_seq=None, teacher_forcing=0.0, max_len=inferred_max_len)
            seq = decode_logits_to_sequence(logits[0], temperature)
            print(f"[{i+1}] {seq}")
            sequences.append(seq)

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Generate sequences from a trained latent decoder.")
    parser.add_argument("--model", required=True, help="Path to trained decoder checkpoint (.pt)")
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--model-type", choices=["gru", "mlp"], default="gru")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--latents", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="default")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Sampling on device: {device}")

    sequences = sample_sequences(
        model_path=args.model,
        latent_dim=args.latent_dim,
        model_type=args.model_type,
        n_samples=args.n,
        temperature=args.temperature,
        device=str(device),
        max_len=args.max_len,
        latents_path=args.latents,
    )

    if args.out:
        outdir = Path("samples") / args.experiment
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / args.out
        with open(outpath, "w") as f:
            for s in sequences:
                f.write(s + "\n")
        print(f"[INFO] Saved {len(sequences)} sequences to {outpath}")


if __name__ == "__main__":
    main()

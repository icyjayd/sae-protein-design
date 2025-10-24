# sae/decoder/cli.py
import argparse
import torch
from transformers import EsmForMaskedLM, EsmTokenizer
from mod_man_utils import add_module
add_module("interplm")
from interplm.sae.inference import load_sae_from_hf
from .data import LatentSequenceDataset, make_dataloader, load_or_create_splits
from .train import train_decoder

def main():
    parser = argparse.ArgumentParser(description="Train latent→sequence decoder.")
    parser.add_argument("--input", required=True, help="CSV, NPY, or TXT file with sequences.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", choices=["gru", "mlp"], default="gru", help="Decoder architecture type.")
    parser.add_argument("--experiment", type=str, default="default", help="Experiment name for caching and outputs.")
    parser.add_argument("--outdir", type=str, default="decoder_out", help="Output directory for checkpoints.")
    parser.add_argument("--causal", action="store_true", help="Use causal mask (autoregressive mode)")

    args = parser.parse_args()

    # --- Load sequences and cached splits ---
    train_seqs, test_seqs, used_cache = load_or_create_splits(args.input, args.experiment)
    print(f"[INFO] Train: {len(train_seqs)} | Test: {len(test_seqs)} | Cached: {used_cache}")

    # --- Load pretrained ESM + SAE models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    esm_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).eval()
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    sae_model = load_sae_from_hf("esm2-8m", plm_layer=6).to(device).eval()

    # --- Build latent datasets ---
    train_ds = LatentSequenceDataset(train_seqs, sae_model, esm_model, tokenizer, device=str(device))
    test_ds = LatentSequenceDataset(test_seqs, sae_model, esm_model, tokenizer, device=str(device))

    # --- Get latent dimension ---
    latent_dim = next(sae_model.parameters()).shape[0] if hasattr(sae_model, "decoder") else 10240
    print(f"[INFO] Latent dim: {latent_dim}")

    # --- Train model ---
    decoder = train_decoder(
        dataset=train_ds,
        latent_dim=latent_dim,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        outdir=args.outdir,
        experiment=args.experiment,
        device=str(device),
        causal=args.causal
    )

    # --- Evaluate on test split ---
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss(ignore_index=-100)
    from .data import make_dataloader
    loader = make_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for latents, tokens in loader:
            latents, tokens = latents.to(device), tokens.to(device)
            logits = decoder(latents, target_seq=tokens)
            loss = criterion(logits.transpose(1, 2), tokens)
            total_loss += loss.item()
    print(f"[RESULT] Test loss = {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    main()

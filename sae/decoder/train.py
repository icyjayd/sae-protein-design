# sae/decoder/train.py
import torch
import torch.nn as nn
from pathlib import Path
from .models import build_decoder
from .data import make_dataloader

import json
from tqdm import tqdm
from sae.utils.esm_utils import perturb_and_decode_whole
from sae.utils.grade_reconstructions import grade_pair


def train_decoder(
    dataset,
    latent_dim: int,
    model_type: str = "gru",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    outdir: str = "decoder_out",
    experiment: str = "default",
    device: str = "cpu",
    resume_from: str = None,
    causal: bool = True,
    eval_every: int = 0,        # 🔥 new
    eval_dataset=None,          # 🔥 new
    sae_model=None,             # 🔥 new (needed for decoding)
    tokenizer=None,             # 🔥 new
):
    """
    Trains the latent→sequence decoder and optionally evaluates it every n epochs.
    """

    outdir = Path(outdir) / experiment
    outdir.mkdir(parents=True, exist_ok=True)

    model = build_decoder(model_type, latent_dim, causal=causal).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[INFO] Starting training: {model_type.upper()} | {len(dataset)} samples | {epochs} epochs")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for latents, tokens in loader:
            latents, tokens = latents.to(device), tokens.to(device)
            optimizer.zero_grad()
            logits = model(latents, target_seq=tokens)
            loss = criterion(logits.transpose(1, 2), tokens)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[{experiment}] Epoch {epoch:02d}/{epochs} | Loss = {avg_loss:.4f}")

        # 🔥 Save checkpoint
        ckpt_path = outdir / f"{model_type}_epoch{epoch}.pt"
        torch.save({"epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict()}, ckpt_path)

        # 🔥 Periodic evaluation
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == epochs):
            if eval_dataset is not None and sae_model is not None and tokenizer is not None:
                print(f"\n[INFO] Running evaluation at epoch {epoch}...")
                eval_results = evaluate_decoder(
                    model=model,
                    dataset=eval_dataset,
                    sae_model=sae_model,
                    tokenizer=tokenizer,
                    device=device,
                    max_sequences=50,
                    outdir=outdir,
                    experiment=f"{experiment}_epoch{epoch}",
                )
                # Optional W&B logging
                try:
                    import wandb
                    wandb.log({"epoch": epoch, **eval_results})
                except ImportError:
                    pass
                print(f"[INFO] Evaluation done at epoch {epoch}\n")

    print(f"[INFO] Training complete. Checkpoints in {outdir}")
    return model

def evaluate_decoder(
    model,
    dataset,
    sae_model=None,
    tokenizer=None,
    batch_size=32,
    device="cpu",
    max_sequences=100,
    outdir="decoder_out",
    experiment="default",
):
    """
    Evaluates a trained decoder quantitatively (loss) and qualitatively
    (decoded sequence similarity via grade_pair).

    Args:
        model: Trained latent→sequence decoder.
        dataset: LatentSequenceDataset containing (latent, token) pairs.
        sae_model: Trained SAE model for latent encoding (required for decoding).
        tokenizer: ESM tokenizer for ID↔AA conversion.
        batch_size: Number of samples per loss batch.
        device: "cpu" or "cuda".
        max_sequences: Number of sequences to decode & score qualitatively.
        outdir: Directory to save evaluation results.
        experiment: Experiment name for results file separation.

    Returns:
        Dict with average loss and reconstruction metrics.
    """
    assert sae_model is not None and tokenizer is not None, \
        "evaluate_decoder requires sae_model and tokenizer."

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    outdir = Path(outdir) / experiment
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # 1️⃣ Quantitative evaluation: cross-entropy loss
    # ------------------------------
    print(f"[INFO] Computing average loss on {len(dataset)} samples...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    with torch.no_grad():
        for latents, tokens in tqdm(loader, desc="Loss Eval"):
            latents, tokens = latents.to(device), tokens.to(device)
            logits = model(latents, target_seq=tokens)
            loss = criterion(logits.transpose(1, 2), tokens)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[EVAL] Average reconstruction loss (CrossEntropy): {avg_loss:.4f}")

    # ------------------------------
    # 2️⃣ Qualitative evaluation: decoded sequence similarity
    # ------------------------------
    print(f"[INFO] Decoding and scoring up to {max_sequences} sequences...")
    subset = [dataset[i] for i in range(min(max_sequences, len(dataset)))]
    results = []
    total_scores = {"identity": 0, "similarity": 0, "norm_align": 0,
                    "lev_sim": 0, "final_score": 0}

    for i, (latents, tokens) in enumerate(tqdm(subset, desc="Decoding")):
        # --- Original sequence reconstruction from tokens ---
        token_ids = tokens.cpu().tolist()
        orig_seq = "".join(
            [tokenizer.convert_ids_to_tokens([tid])[0]
             for tid in token_ids if tid not in tokenizer.all_special_ids]
        )

        # --- Decode from latent using the trained decoder ---
        latents = latents.unsqueeze(0).to(device)
        with torch.no_grad():
            decoded_seq = perturb_and_decode_whole(
                sequence=orig_seq,
                sae_model=sae_model,
                decoder=model,
                tokenizer=tokenizer,
                latent_deltas={},  # no perturbation → pure reconstruction
                device=device,
            )

        # --- Compute similarity metrics ---
        scores = grade_pair(orig_seq, decoded_seq)
        results.append({
            "index": i,
            "original": orig_seq,
            "decoded": decoded_seq,
            **scores
        })

        # Aggregate averages
        for k in total_scores:
            total_scores[k] += scores.get(k, 0.0)

    n = len(results)
    avg_scores = {k: v / n for k, v in total_scores.items()}

    # ------------------------------
    # 3️⃣ Summaries and saving
    # ------------------------------
    print("\n[EVAL] Reconstruction quality (averaged):")
    print(f"  {'loss':12s} : {avg_loss:.4f}")
    for k, v in avg_scores.items():
        print(f"  {k:12s} : {v:.4f}")

    # Save full report
    out_path = outdir / f"decoder_eval_{experiment}.json"
    with open(out_path, "w") as f:
        json.dump({
            "average_loss": avg_loss,
            "average_scores": avg_scores,
            "samples": results
        }, f, indent=2)

    print(f"[INFO] Saved detailed evaluation results to {out_path}")

    return {"loss": avg_loss, **avg_scores}

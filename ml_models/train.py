# sae/decoder/train.py
import torch
import torch.nn as nn
from pathlib import Path
from .models import build_decoder
from .data import make_dataloader


def _get_latest_checkpoint(experiment_dir: Path, model_type: str):
    """Returns the latest checkpoint for a given model type, or None."""
    ckpts = sorted(experiment_dir.glob(f"{model_type}_epoch*.pt"))
    if not ckpts:
        return None
    latest = ckpts[-1]
    print(f"[INFO] Found existing checkpoint: {latest.name}")
    return latest


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
):
    """
    Trains the latent→sequence decoder (GRU or MLP) using the provided dataset.
    Supports resuming from previous checkpoints automatically or manually.
    """
    outdir = Path(outdir) / experiment
    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Build model
    # ----------------------------------------------------------
    model = build_decoder(model_type, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ----------------------------------------------------------
    # Resume logic
    # ----------------------------------------------------------
    start_epoch = 0
    if resume_from:
        ckpt_path = Path(resume_from)
        print(f"[INFO] Resuming training from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0)
    else:
        latest = _get_latest_checkpoint(outdir, model_type)
        if latest:
            print(f"[INFO] Automatically resuming from {latest.name}")
            checkpoint = torch.load(latest, map_location=device)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                start_epoch = checkpoint.get("epoch", 0)
            else:
                print("[WARN] Old-format checkpoint detected (no optimizer). Starting fresh.")

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[INFO] Starting training: {model_type.upper()} | {len(dataset)} samples | {epochs} epochs total")
    print(f"[INFO] Checkpoints directory: {outdir}")
    print(f"[INFO] Resuming at epoch {start_epoch + 1}")

    for epoch in range(start_epoch + 1, epochs + 1):
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

        # Save checkpoint (includes optimizer)
        ckpt_path = outdir / f"{model_type}_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_path,
        )

    print(f"[INFO] Training complete. Checkpoints saved under {outdir}")
    return model


def evaluate_decoder(model, dataset, batch_size=32, device="cpu"):
    """
    Evaluates a trained decoder on a dataset.
    Returns average cross-entropy loss.
    """
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for latents, tokens in loader:
            latents, tokens = latents.to(device), tokens.to(device)
            logits = model(latents, target_seq=tokens)
            loss = criterion(logits.transpose(1, 2), tokens)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[EVAL] Test loss = {avg_loss:.4f}")
    return avg_loss

# sae/decoder/train.py
import torch
import torch.nn as nn
from pathlib import Path
from .models import build_decoder
from .data import make_dataloader


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
    **kwargs
):
    """
    Trains the latent→sequence decoder (GRU or MLP) using the provided dataset.

    Args:
        dataset: LatentSequenceDataset with (latent, token) pairs
        latent_dim: SAE latent dimensionality
        model_type: "gru" or "mlp"
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        outdir: Directory to save checkpoints
        experiment: Experiment name for subfolder separation
        device: "cpu" or "cuda"
    Returns:
        The trained model instance
    """
    outdir = Path(outdir) / experiment
    outdir.mkdir(parents=True, exist_ok=True)

    # Build model and optimizer
    causal = kwargs.pop("causal", True)
    model = build_decoder(model_type, latent_dim, causal=causal).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # DataLoader
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=True)

    print(f"[INFO] Starting training: {model_type.upper()} | {len(dataset)} samples | {epochs} epochs")
    print(f"[INFO] Checkpoints: {outdir}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for latents, tokens in loader:
            latents, tokens = latents.to(device), tokens.to(device)
            optimizer.zero_grad()

            # Forward
            logits = model(latents, target_seq=tokens)
            loss = criterion(logits.transpose(1, 2), tokens)

            # Backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[{experiment}] Epoch {epoch:02d}/{epochs} | Loss = {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = outdir / f"{model_type}_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)

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

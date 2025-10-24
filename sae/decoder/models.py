# sae/decoder/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. GRU Decoder (Autoregressive)
# =========================================================
class LatentDecoderGRU(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512, vocab_size=20, num_layers=2):
        super().__init__()
        # FIX: project latent into all GRU layers' hidden states
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(vocab_size, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_vocab = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, z, target_seq=None, teacher_forcing=0.5, max_len=None):
        """
        z: (B, latent_dim)
        target_seq: (B, L) indices of true tokens
        """
        B = z.size(0)
        # FIX: reshape into (num_layers, B, hidden_dim)
        h = self.latent_to_hidden(z).view(self.num_layers, B, self.hidden_dim)

        x = torch.zeros(B, 1, self.vocab_size, device=z.device)
        outputs = []
        L = target_seq.size(1) if target_seq is not None else max_len or 256

        for t in range(L):
            out, h = self.gru(x, h)
            logits = self.hidden_to_vocab(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing logic
            if target_seq is not None and torch.rand(1).item() < teacher_forcing:
                idx = target_seq[:, t].clamp(min=0)
                x = F.one_hot(idx, num_classes=self.vocab_size).float().unsqueeze(1)
            else:
                x = F.one_hot(logits.argmax(-1), num_classes=self.vocab_size).float().unsqueeze(1)

        return torch.cat(outputs, dim=1)


# =========================================================
# 2. MLP Decoder (One-shot)
# =========================================================
class LatentDecoderMLP(nn.Module):
    """
    Maps a latent vector directly to a fixed-length amino acid sequence.
    Now dynamically crops output if target sequence is shorter.
    """
    def __init__(self, latent_dim, hidden_dim=1024, vocab_size=20, max_len=256):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_len * vocab_size)
        )

    def forward(self, z, target_seq=None, **_):
        # FIX: dynamically match sequence length to target if provided
        out = self.model(z)
        out = out.view(z.size(0), self.max_len, self.vocab_size)
        if target_seq is not None and target_seq.size(1) < self.max_len:
            out = out[:, : target_seq.size(1), :]
        return out


# =========================================================
# 3. Factory
# =========================================================
def build_decoder(model_type: str, latent_dim: int, **kwargs) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "gru":
        print("[INFO] Using GRU decoder.")
        return LatentDecoderGRU(latent_dim, **kwargs)
    elif model_type == "mlp":
        print("[INFO] Using MLP decoder.")
        return LatentDecoderMLP(latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

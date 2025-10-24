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
# 3. Transformer Decoder (latent-conditioned, causal or bidirectional)
# =========================================================

class LatentTransformerDecoder(nn.Module):
    """
    Transformer-based decoder that conditions on a global SAE latent vector.

    Modes:
      - causal=True:   autoregressive (each token attends only to past tokens)
      - causal=False:  bidirectional (each token attends to all tokens, for reconstruction)
    """

    def __init__(
        self,
        latent_dim,
        vocab_size=20,
        d_model=512,
        n_heads=8,
        num_layers=6,
        max_len=256,
        dropout=0.1,
        causal=True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.causal = causal

        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.latent_proj.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def _generate_causal_mask(self, size, device):
        # Upper-triangular mask to block attention to future tokens
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, z, target_seq=None, teacher_forcing=0.0, max_len=None):
        """
        z: (B, latent_dim)
        target_seq: (B, L) token indices
        """
        B = z.size(0)
        device = z.device
        L = target_seq.size(1) if target_seq is not None else max_len or self.max_len

        latent_context = self.latent_proj(z).unsqueeze(1)  # (B, 1, d_model)
        tok_emb = self.token_embed(target_seq) + self.pos_embed[:, :L, :]
        tok_emb = self.dropout(tok_emb)

        # causal mask for autoregressive training
        tgt_mask = self._generate_causal_mask(L, device) if self.causal else None

        out = self.transformer(tgt=tok_emb, memory=latent_context, tgt_mask=tgt_mask)
        logits = self.output_layer(out)
        return logits


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
    elif model_type == "transformer":
        causal = kwargs.pop("causal", True)
        print(f"[INFO] Using Transformer decoder (causal={causal}).")
        return LatentTransformerDecoder(latent_dim, causal=causal, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

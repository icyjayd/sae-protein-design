import torch
import torch.nn as nn

class LatentFieldPOC(nn.Module):
    """
    Simple proof-of-concept model that learns how much to edit each latent feature
    (per position, per latent dimension) based on a requested magnitude m.
    """

    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        # Map the requested magnitude (scalar) to a latent-direction vector
        self.m_to_emb = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        # Learnable per-latent scale
        self.scale = nn.Parameter(torch.ones(latent_dim))

    def forward(self, Z: torch.Tensor, m: torch.Tensor):
        """
        Args:
            Z:  (L, N) tensor of SAE latent activations
            m:  scalar tensor representing desired magnitude of property change

        Returns:
            dZ: (L, N) tensor of latent deltas (same shape as Z)
        """
        base_delta = self.m_to_emb(m.unsqueeze(0))  # (1, N)
        dZ = base_delta * self.scale * 0.1
        dZ = dZ.expand(Z.shape[0], -1)  # same length as sequence
        return dZ

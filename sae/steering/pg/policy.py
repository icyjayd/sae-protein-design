import torch
import torch.nn as nn
from torch.distributions import Normal

class StochasticLatentPolicy(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128, sigma: float = 0.1):
        super().__init__()
        self.m_to_emb = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim)
        )
        self.scale = nn.Parameter(torch.ones(latent_dim))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
    def forward(self, Z: torch.Tensor, m: torch.Tensor):
        base = self.m_to_emb(m.view(1,1))
        dZ_mean = (base.squeeze(0) * self.scale * 0.1).expand(Z.shape[0], -1)
        return dZ_mean
    def sample(self, Z: torch.Tensor, m: torch.Tensor):
        dZ_mean = self.forward(Z, m)
        dist = Normal(dZ_mean, self.sigma)
        dZ = dist.rsample()
        logp = dist.log_prob(dZ).sum()
        ent = dist.entropy().sum()
        return dZ, dZ_mean, logp, ent

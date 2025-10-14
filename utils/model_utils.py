
import torch
import torch.nn as nn
import numpy as np

def get_device(pref: str = None):
    if pref is not None and pref.strip():
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparseAutoencoderSAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=64, hidden=512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.encode(x); out = self.decode(z); return out, z

class MonosemanticSAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=64, hidden=512, topk=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.topk = topk
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, input_dim, bias=True)
        )
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.decoder[0].weight, a=np.sqrt(5))

    def encode(self, x):
        z = self.encoder(x)
        if self.topk is not None and self.topk > 0:
            k = min(self.topk, z.shape[1])
            _, topk_idx = torch.topk(torch.abs(z), k=k, dim=1)
            mask = torch.zeros_like(z); mask.scatter_(1, topk_idx, 1.0)
            z = z * mask
        return z

    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.encode(x); out = self.decode(z); return out, z

    def latent_decorrelation_loss(self, z, eps=1e-6):
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (zc.shape[0] + eps)
        diag = torch.diag(torch.diag(cov)); off = cov - diag
        return (off ** 2).mean()

    def decoder_orthonormal_loss(self):
        W = self.decoder[0].weight; G = W.T @ W
        I = torch.eye(G.shape[0], device=G.device)
        return ((G - I) ** 2).mean()

    def decoder_unitnorm_loss(self, target=1.0, eps=1e-6):
        W = self.decoder[0].weight
        col_norms = torch.sqrt(torch.sum(W**2, dim=0) + eps)
        return ((col_norms - target) ** 2).mean()

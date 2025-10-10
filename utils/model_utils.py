\"\"\"Defines a simple SAE tailored to activations; encoder + decoder symmetric MLPs.\"\"\"
import torch
import torch.nn as nn

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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z

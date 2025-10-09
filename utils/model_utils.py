\"\"\"Model utilities: minimal sparse autoencoder and training loop.\"\"\"
import torch
import torch.nn as nn
import numpy as np

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=1000, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(64, latent_dim*8)),
            nn.ReLU(),
            nn.Linear(max(64, latent_dim*8), latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(64, latent_dim*8)),
            nn.ReLU(),
            nn.Linear(max(64, latent_dim*8), input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z

def train_autoencoder(model, X, device='cpu', epochs=20, batch_size=32, lr=1e-3, sparsity_coef=1e-3, out_dir=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    X_t = torch.from_numpy(X).to(device)
    n = X.shape[0]
    for ep in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i+batch_size]
            xb = X_t[batch_idx]
            optimizer.zero_grad()
            recon, z = model(xb)
            # reshape for softmax-like interpretation: using MSE on probabilities is acceptable here
            recon_probs = recon.view(recon.size(0), -1)
            loss_recon = mse(recon_probs, xb.view(recon.size(0), -1))
            loss_sparsity = sparsity_coef * torch.mean(torch.abs(z))
            loss = loss_recon + loss_sparsity
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / n
        if (ep+1) % 5 == 0 or ep==0:
            print(f"Epoch {ep+1}/{epochs} avg_loss={avg:.6f}")
    if out_dir is not None:
        torch.save(model.state_dict(), out_dir / "sae_model.pt")
        print("Saved model to", out_dir / "sae_model.pt")

def decode_latent_batch(model, z_tensor):
    \"\"\"Given a PyTorch model and a z vector (1,Nlat), produce a rough sequence string by decoding and argmaxing.\"\"\"
    model.eval()
    with torch.no_grad():
        if isinstance(z_tensor, torch.Tensor):
            z = z_tensor.float()
        else:
            z = torch.tensor(z_tensor, dtype=torch.float32)
        out = model.decode(z)
        arr = out.cpu().numpy()
    # arr corresponds to flattened one-hot logits; convert to sequence by argmax per position
    n = arr.shape[0]
    LA = arr.shape[1]
    # assume alphabet size 20 provided by config in file usage; fallback to 20
    A = 20
    L = LA // A
    seqs = []
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for i in range(n):
        mat = arr[i].reshape(L, A)
        idxs = mat.argmax(axis=1)
        seqs.append(''.join(alphabet[k] for k in idxs))
    return np.array(seqs)

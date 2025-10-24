"""
Tests training and sampling logic using dummy data and small models.
"""
import torch
import numpy as np
from sae.decoder.train import train_decoder
from sae.decoder.sample import sample_sequences

class DummyDataset(torch.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        latent = torch.randn(8)
        seq = torch.randint(0, 5, (6,))
        return latent, seq

def test_train_decoder(tmp_path):
    ds = DummyDataset()
    model = train_decoder(ds, latent_dim=8, model_type="mlp", epochs=1, batch_size=2, outdir=tmp_path, experiment="unit", device="cpu")
    assert any(tmp_path.glob("unit/*.pt"))

def test_sampling(tmp_path):
    from sae.decoder.models import LatentDecoderMLP
    m = LatentDecoderMLP(latent_dim=8, vocab_size=5)
    path = tmp_path / "m.pt"
    torch.save(m.state_dict(), path)
    seqs = sample_sequences(model_path=path, latent_dim=8, model_type="mlp", n_samples=2, device="cpu", max_len=10)
    assert len(seqs) == 2
    assert all(isinstance(s, str) for s in seqs)

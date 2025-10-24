"""
Tests decoder model definitions (GRU and MLP).
"""
import torch
from sae.decoder.models import LatentDecoderGRU, LatentDecoderMLP, build_decoder

def test_gru_forward_pass():
    model = LatentDecoderGRU(latent_dim=8, hidden_dim=16, vocab_size=5)
    z = torch.randn(2, 8)
    out = model(z, target_seq=torch.randint(0, 5, (2, 10)))
    assert out.shape == (2, 10, 5)

def test_mlp_forward_pass():
    model = LatentDecoderMLP(latent_dim=8, hidden_dim=16, vocab_size=5, max_len=12)
    z = torch.randn(2, 8)
    out = model(z)
    assert out.shape == (2, 12, 5)

def test_model_factory():
    gru = build_decoder("gru", latent_dim=4)
    mlp = build_decoder("mlp", latent_dim=4)
    assert isinstance(gru, LatentDecoderGRU)
    assert isinstance(mlp, LatentDecoderMLP)

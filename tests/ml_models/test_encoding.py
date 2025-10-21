import numpy as np, torch, pytest
from ml_models import encoding

def test_aac_encoding():
    seqs = ["ACDE", "GGGG"]
    X = encoding.encode_aac(seqs)
    assert X.shape == (2, 20)
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-5)

def test_kmer_encoding():
    seqs = ["ACDEFG", "GGGGGG"]
    X = encoding.encode_kmer(seqs, k=2)
    assert X.shape[0] == 2
    assert np.isfinite(X).all()

def test_onehot_encoding():
    seqs = ["ACDE", "GGGG"]
    X = encoding.encode_onehot(seqs, max_len=8)
    assert X.shape == (2, 8 * 20)
    assert np.all((X == 0) | (X == 1))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optional ESM test")
def test_esm_encoding(monkeypatch):
    def fake_forward(*a, **kw):
        class FakeOut: last_hidden_state = torch.zeros(1, 8, 64)
        return FakeOut()
    monkeypatch.setattr("transformers.EsmModel.from_pretrained",
        lambda *a, **k: type("Fake", (), {"eval": lambda s: s, "to": lambda s,d: s, "__call__": fake_forward})())
    monkeypatch.setattr("transformers.EsmTokenizer.from_pretrained",
        lambda *a, **k: lambda s, **kw: {"input_ids": torch.zeros(1, 8, dtype=torch.long)})
    X = encoding.encode_esm(["ACDE"], device="cpu")
    assert X.ndim == 2

"""
Tests data loading, caching, and dataset creation for the decoder pipeline.
"""
import torch
import pandas as pd
from sae.decoder.data import load_or_create_splits, LatentSequenceDataset, make_dataloader, get_cache_dir


class DummySAE:
    def encode(self, x): return x.mean(dim=0, keepdim=True)
class DummyESM:
    def __call__(self, **_): return type("O", (), {"hidden_states": [torch.randn(1, 5, 16)]})
class DummyTokenizer:
    def __call__(self, seq, **_): return {"input_ids": torch.arange(5).unsqueeze(0)}


def test_load_or_create_splits_csv(tmp_path):
    csv_path = tmp_path / "seqs.csv"
    df = pd.DataFrame({"sequence": ["AAA", "BBB", "CCC", "DDD"], "split": ["train", "test", "train", "test"]})
    df.to_csv(csv_path, index=False)

    # FIX: clear cache for clean test
    cache_dir = get_cache_dir("unit_exp")
    for f in cache_dir.glob("*"):
        f.unlink()

    train, test, cached = load_or_create_splits(csv_path, "unit_exp")
    assert len(train) == 2 and len(test) == 2
    assert not cached


def test_dataset_creation_and_dataloader(tmp_path):
    seqs = ["ACDE", "FGHI"]
    sae, esm, tok = DummySAE(), DummyESM(), DummyTokenizer()
    ds = LatentSequenceDataset(seqs, sae, esm, tok)
    latents, tokens = ds[0]
    assert isinstance(latents, torch.Tensor)
    assert tokens.ndim == 1
    dl = make_dataloader(ds, batch_size=2)
    batch = next(iter(dl))
    assert isinstance(batch, tuple)
    assert batch[0].shape[0] == 2

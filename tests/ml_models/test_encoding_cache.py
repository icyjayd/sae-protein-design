import os
import json
import numpy as np
import pytest

from ml_models.encoding import (
    encode_sequences,
    encode_aac,
    encode_dpc,
    encode_kmer,
    encode_onehot,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="function")
def seqs():
    """Small synthetic dataset of valid amino acid sequences."""
    return [
        "ACDEFGHIKLMNPQRSTVWY",
        "AAAAAAAAAAAAAAAAAAAA",
        "CCCCCCCCCCCCCCCCCCCC",
        "DEFGHIKLMNPQRSTVWYAC",
        "MNPQRSTVWYACDEFGHIKL",
    ]


@pytest.fixture(scope="function")
def cache_dir(tmp_path_factory):
    """Unique cache folder per test to prevent collisions."""
    path = tmp_path_factory.mktemp("cache_test")
    return str(path)


# ---------------------------------------------------------------------
# Basic encoder correctness
# ---------------------------------------------------------------------
def test_aac_dim(seqs):
    arr = encode_aac(seqs)
    assert arr.shape[1] == 20
    assert np.allclose(arr.sum(axis=1), 1.0, atol=1e-6)


def test_dpc_dim(seqs):
    arr = encode_dpc(seqs)
    assert arr.shape[1] == 400
    assert np.all(arr.sum(axis=1) > 0)


def test_kmer_dim(seqs):
    arr = encode_kmer(seqs, k=3)
    assert arr.shape[1] == 8000


def test_onehot_shape(seqs):
    arr = encode_onehot(seqs, max_len=10)
    assert arr.shape == (len(seqs), 10 * 20)
    assert np.all((arr == 0) | (arr == 1))


# ---------------------------------------------------------------------
# Caching behavior
# ---------------------------------------------------------------------
def test_cache_creation_and_reuse(seqs, cache_dir):
    """Ensure cache file and metadata are created and reused."""
    arr1 = encode_sequences(seqs, encoding="aac", cache_base=cache_dir)
    cache_path = os.path.join(cache_dir, "aac")
    npy_files = [f for f in os.listdir(cache_path) if f.endswith(".npy")]
    meta_files = [f for f in os.listdir(cache_path) if f.endswith(".meta.json")]
    assert len(npy_files) == len(meta_files) == 1

    arr2 = encode_sequences(seqs, encoding="aac", cache_base=cache_dir)
    assert np.allclose(arr1, arr2)


def test_subset_load(seqs, cache_dir):
    """Cache larger, then request subset and ensure partial load."""
    seqs_large = seqs * 4  # 20 sequences total
    arr_full = encode_sequences(seqs_large, encoding="aac", cache_base=cache_dir)
    n_full = len(arr_full)

    subset = seqs_large[:5]
    arr_sub = encode_sequences(subset, encoding="aac", cache_base=cache_dir)
    assert np.allclose(arr_sub, arr_full[:5])
    assert len(arr_full) == n_full


def test_cache_expansion(seqs, cache_dir):
    """Cache smaller, then request larger → recompute and create new cache entry."""
    # First: cache small subset
    subset = seqs[:2]
    arr_small = encode_sequences(subset, encoding="aac", cache_base=cache_dir)
    meta_dir = os.path.join(cache_dir, "aac")
    meta_files_before = [f for f in os.listdir(meta_dir) if f.endswith(".meta.json")]
    meta_before_path = os.path.join(meta_dir, meta_files_before[0])
    n_before = json.load(open(meta_before_path))["n_sequences"]

    # Encode larger dataset with new sequences
    seqs_large = seqs + [
        "VVVVVVVVVVVVVVVVVVVV",
        "YYYYYYYYYYYYYYYYYYYY",
        "WWWWWWWWWWWWWWWWWWWW",
    ]
    arr_large = encode_sequences(seqs_large, encoding="aac", cache_base=cache_dir)

    # Locate newest meta file
    meta_files_after = [f for f in os.listdir(meta_dir) if f.endswith(".meta.json")]
    latest_meta = max(
        (os.path.join(meta_dir, f) for f in meta_files_after),
        key=os.path.getmtime,
    )
    n_after = json.load(open(latest_meta))["n_sequences"]

    assert arr_small.shape[0] == len(subset)
    assert arr_large.shape[0] == len(seqs_large)
    assert n_after > n_before


def test_no_cache_option(seqs, cache_dir):
    """Disabling cache should not write files."""
    arr = encode_sequences(seqs, encoding="aac", use_cache=False, cache_base=cache_dir)
    assert not os.path.exists(os.path.join(cache_dir, "aac"))
    assert arr.shape[1] == 20


def test_different_encodings_isolate_cache(seqs, cache_dir):
    """Different encodings should have separate cache dirs."""
    encode_sequences(seqs, encoding="aac", cache_base=cache_dir)
    encode_sequences(seqs, encoding="dpc", cache_base=cache_dir)
    dirs = os.listdir(cache_dir)
    assert "aac" in dirs and "dpc" in dirs
    assert any(f.endswith(".npy") for f in os.listdir(os.path.join(cache_dir, "aac")))
    assert any(f.endswith(".npy") for f in os.listdir(os.path.join(cache_dir, "dpc")))


def test_subset_vs_full_content_identical(seqs, cache_dir):
    """Ensure that cached subset matches full prefix numerically."""
    full_arr = encode_sequences(seqs * 3, encoding="aac", cache_base=cache_dir)
    sub_arr = encode_sequences(seqs, encoding="aac", cache_base=cache_dir)
    assert np.allclose(full_arr[:len(seqs)], sub_arr)

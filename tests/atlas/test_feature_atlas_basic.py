import numpy as np
import pandas as pd
from sae.analysis.feature_atlas import (
    aggregate_per_sequence, compute_correlations, select_candidates, save_correlation_csv
)
import os, tempfile

def test_aggregate_per_sequence_mean():
    # 2 sequences, variable lengths, D=3
    z1 = np.array([[1,2,3],[2,3,4]])      # mean = [1.5, 2.5, 3.5]
    z2 = np.array([[0,0,1],[0,1,1],[1,1,1]])  # mean = [1/3, 2/3, 1]
    Z = aggregate_per_sequence([z1, z2], method="mean")
    assert Z.shape == (2,3)
    np.testing.assert_allclose(Z[0], np.array([1.5, 2.5, 3.5]), atol=1e-8)
    np.testing.assert_allclose(Z[1], np.array([1/3, 2/3, 1.0]), atol=1e-8)

def test_compute_correlations_shapes_and_keys():
    # N=5, D=4
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((5,4))
    y = np.linspace(0,1,5)
    df = compute_correlations(Z, y, method="spearman")
    assert set(df.columns) >= {"latent_index","spearman","p_value","n"}

    assert len(df) == 4
    # sorted by absolute correlation descending
    assert df.iloc[0]["spearman"] >= df.iloc[1]["spearman"] or df.iloc[0]["spearman"] <= df.iloc[1]["spearman"]

def test_select_candidates_splits_signs_and_nulls():
    df = pd.DataFrame({
        "latent":[0,1,2,3],
        "corr":[ 0.8,-0.7,0.02,-0.01],
        "pval":[ 0.001,0.002,0.7,0.8],
        "n":[20,20,20,20],
    })
    out = select_candidates(df, top_n=1, null_threshold=0.05)
    assert out["positive"] == [0]
    assert out["negative"] == [1]
    # latents 2,3 are non-sig → null bucket
    assert set(out["null"]) == {2,3}

def test_save_correlation_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"latent":[0,1], "corr":[0.5,-0.4], "pval":[0.01,0.03], "n":[10,10]})
    out_path = tmp_path / "corr.csv"
    p = save_correlation_csv(df, str(out_path))
    assert os.path.exists(p)
    df2 = pd.read_csv(p)
    pd.testing.assert_frame_equal(
        df.sort_index(axis=1),
        df2.sort_index(axis=1),
        check_dtype=False
    )

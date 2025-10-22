import numpy as np
import pandas as pd

from ml_models.data_utils import load_data


def test_load_data_npy_sequences_npy_labels(tmp_path):
    # Prepare .npy sequences and labels
    seqs = np.array([
        "MKTLLILAVITAIAAGALA",
        "ACDEFGHIKLMNPQRSTVWY",
    ], dtype=object)
    labels = np.array([0.1, 0.9], dtype=float)

    seq_path = tmp_path / "seqs.npy"
    label_path = tmp_path / "labels.npy"
    np.save(seq_path, seqs)
    np.save(label_path, labels)

    df = load_data(str(seq_path), str(label_path))
    assert set(df.columns) == {"sequence", "label"}
    assert len(df) == 2
    # Sequences normalized to valid amino acids
    assert df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+").all()
    assert np.allclose(df["label"].to_numpy(dtype=float), labels)


def test_load_data_csv_sequences_npy_labels(tmp_path):
    # Prepare CSV sequences (no labels) and .npy labels
    seqs = [
        "MKTLLILAVITAIAAGALA",
        "ACDEFGHIKLMNPQRSTVWY",
        "ACACACACACACACACACAC",
    ]
    df_seq = pd.DataFrame({"sequence": seqs})
    csv_path = tmp_path / "seqs.csv"
    df_seq.to_csv(csv_path, index=False)

    labels = np.array([0.2, 0.8, 0.5], dtype=float)
    label_path = tmp_path / "labels.npy"
    np.save(label_path, labels)

    df = load_data(str(csv_path), str(label_path))
    assert set(df.columns) == {"sequence", "label"}
    assert len(df) == 3
    assert df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+").all()
    assert np.allclose(df["label"].to_numpy(dtype=float), labels)


def test_load_data_npy_sequences_csv_labels(tmp_path):
    # Prepare .npy sequences and CSV labels (row-aligned)
    seqs = np.array([
        "MKVVVVVVVVVVVVVVVVV",
        "ACACACACACACACACACAC",
        "MKTLLILAVITAIAAGALA",
    ], dtype=object)
    seq_path = tmp_path / "seqs.npy"
    np.save(seq_path, seqs)

    labels = pd.DataFrame({"label": [0.3, 0.6, 0.9]})
    label_csv = tmp_path / "labels.csv"
    labels.to_csv(label_csv, index=False)

    df = load_data(str(seq_path), str(label_csv))
    assert set(df.columns) == {"sequence", "label"}
    assert len(df) == 3
    assert df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+").all()
    assert np.allclose(df["label"].to_numpy(dtype=float), labels["label"].to_numpy())

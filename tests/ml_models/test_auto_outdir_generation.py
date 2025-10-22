import os
import json
import shutil
import uuid
import pandas as pd
import numpy as np
import pytest
from ml_models.train import train_model, _auto_outdir


@pytest.fixture(scope="function")
def synthetic_data(tmp_path):
    """Create small synthetic CSVs for quick training with matching IDs."""
    seqs = ["ACDEFGHIKLMNPQRSTVWY"[:10]] * 10
    labels = np.linspace(0, 1, 10)
    ids = np.arange(len(seqs))

    seq_file = tmp_path / "seqs.csv"
    label_file = tmp_path / "labels.csv"

    # Both files share the same ids so load_data() can merge them
    pd.DataFrame({"id": ids, "sequence": seqs}).to_csv(seq_file, index=False)
    pd.DataFrame({"id": ids, "label": labels}).to_csv(label_file, index=False)

    yield str(seq_file), str(label_file)

@pytest.fixture(scope="function")
def temp_run_dir():
    """Provide a unique test run subdir and clean only it after."""
    unique_prefix = f"test_{uuid.uuid4().hex[:8]}"
    yield unique_prefix
    base_dir = os.path.join("runs", "ml_model")
    if os.path.isdir(base_dir):
        for d in os.listdir(base_dir):
            if d.startswith(unique_prefix):
                shutil.rmtree(os.path.join(base_dir, d), ignore_errors=True)

def test_train_model_with_npys(tmp_path, temp_run_dir):
    """Ensure .npy inputs are accepted and auto-outdir still works."""
    seqs = np.array(["ACDEFGHIKL", "LMNPQRSTVW", "YACDEFGHIK", "LMNPQRSTVW", "YACDEFGHIK"])
    labels = np.linspace(0, 1, len(seqs))
    seq_path = tmp_path / "seqs.npy"
    label_path = tmp_path / "labels.npy"
    np.save(seq_path, seqs)
    np.save(label_path, labels)

    prefix = temp_run_dir
    _ = train_model(
        seq_file=str(seq_path),
        labels_file=str(label_path),
        model_name="rf",
        encoding="aac",
        n_samples=len(seqs),
        prefix=prefix,
    )

    base_dir = os.path.join("runs", "ml_model")
    dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    assert len(dirs) == 1
    run_dir = os.path.join(base_dir, dirs[0])
    assert os.path.isfile(os.path.join(run_dir, "model.pkl"))
    assert os.path.isfile(os.path.join(run_dir, "run.json"))

def test_auto_outdir_helper():
    """Ensure _auto_outdir produces the right folder name."""
    d1 = _auto_outdir("rf", "aac", None)
    d2 = _auto_outdir("rf", "aac", 500)
    d3 = _auto_outdir("rf", "aac", 500, prefix="exp1")
    assert d1.endswith("rf_aac_all")
    assert d2.endswith("rf_aac_500")
    assert d3.endswith("exp1_rf_aac_500")
    assert d1.startswith("runs/ml_model")


def test_train_model_auto_outdir(synthetic_data, temp_run_dir):
    """Check that train_model auto-generates proper folder if outdir not provided."""
    seq_file, label_file = synthetic_data
    prefix = temp_run_dir
    _ = train_model(
        seq_file=seq_file,
        labels_file=label_file,
        model_name="rf",
        encoding="aac",
        n_samples=10,
        prefix=prefix,
    )

    base_dir = os.path.join("runs", "ml_model")
    dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    assert len(dirs) == 1, f"No generated dir found under {base_dir}"
    run_dir = os.path.join(base_dir, dirs[0])

    assert os.path.isfile(os.path.join(run_dir, "model.pkl"))
    assert os.path.isfile(os.path.join(run_dir, "scaler.pkl"))
    run_json = os.path.join(run_dir, "run.json")
    assert os.path.isfile(run_json)

    with open(run_json) as f:
        run_data = json.load(f)
    assert run_data["model"] == "rf"
    assert run_data["encoding"] == "aac"
    assert run_data["n_samples"] == 10


def test_train_model_explicit_outdir(synthetic_data, tmp_path):
    """Check that explicit outdir overrides automatic generation."""
    outdir = tmp_path / "manual_out"
    seq_file, label_file = synthetic_data
    _ = train_model(
        seq_file=seq_file,
        labels_file=label_file,
        model_name="rf",
        encoding="aac",
        outdir=str(outdir),
    )
    assert outdir.exists()
    assert os.path.isfile(outdir / "model.pkl")
    assert os.path.isfile(outdir / "run.json")

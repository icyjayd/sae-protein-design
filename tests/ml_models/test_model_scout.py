import os
import json
import numpy as np
import pytest
from pathlib import Path
from proteng_scout.main import run_scout

@pytest.fixture
def tmp_data(tmp_path):
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC"]
    labels = [0.1, 0.2, 0.3]
    seq_file = tmp_path / "seqs.npy"
    label_file = tmp_path / "labels.npy"
    np.save(seq_file, np.array(seqs))
    np.save(label_file, np.array(labels))
    return seq_file, label_file, tmp_path

def test_run_scout_minimal(tmp_data):
    seq_file, label_file, tmp_path = tmp_data
    outpath = tmp_path / "results.json"
    result = run_scout(
        seq_file=str(seq_file),
        labels_file=str(label_file),
        models=["ridge"],
        encodings=["aac"],
        sample_grid=[3],
        n_jobs=1,
        outpath=str(outpath)
    )
    assert outpath.exists()
    data = json.load(open(outpath))
    assert "ranked_results" in data
    assert isinstance(data["ranked_results"], list)
    assert Path(data["html_report"]).exists()

import pytest
import pandas as pd


def pytest_addoption(parser):
    """Add command-line options for real data testing"""
    parser.addoption(
        "--sequences",
        action="store",
        default=None,
        help="Path to real sequences file (.npy or .csv)"
    )
    parser.addoption(
        "--labels",
        action="store",
        default=None,
        help="Path to real labels file (.npy or .csv)"
    )
    parser.addoption(
        "--n-samples",
        action="store",
        type=int,
        default=None,
        help="Number of samples to use from data (default: use all)"
    )


@pytest.fixture
def tmp_seq_file(tmp_path):
    data = {
        "id": ["a", "b", "c", "d"],
        "sequence": ["MKTLLILAVITAIAAGALA", "ACDEFGHIKLMNPQRSTVWY",
                     "MKVVVVVVVVVVVVVVVVV", "ACACACACACACACACACAC"],
        "label": [0.1, 0.9, 0.2, 0.8]
    }
    df = pd.DataFrame(data)
    seq_path = tmp_path / "seqs.csv"
    df.to_csv(seq_path, index=False)
    return seq_path
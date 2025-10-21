import pytest
import pandas as pd
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

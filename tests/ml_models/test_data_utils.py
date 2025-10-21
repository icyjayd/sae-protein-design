from ml_models import data_utils

def test_load_and_split(tmp_seq_file):
    df = data_utils.load_data(tmp_seq_file)
    assert set(df.columns) == {"sequence", "label"}
    train, test = data_utils.split_data(df, seed=42)
    assert len(train) + len(test) == len(df)

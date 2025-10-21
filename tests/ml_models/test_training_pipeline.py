import joblib
from ml_models.train import train_model
from pathlib import Path

def test_train_model(tmp_seq_file, tmp_path):
    out = tmp_path / "run"
    metrics = train_model(seq_file=tmp_seq_file, model_name="rf", encoding="aac",
                          task="regression", seed=123, outdir=out)
    assert isinstance(metrics, dict)
    assert (out / "model.pkl").exists()
    assert (out / "scaler.pkl").exists()
    model = joblib.load(out / "model.pkl")
    assert hasattr(model, "predict")

import pandas as pd
from proteng_scout.run_single import run_single

def test_run_single_basic(monkeypatch):
    df = pd.DataFrame({
        "sequence": ["ACDEFGHIKL", "AAAAAAAAAA", "CCCCCCCCCC"],
        "label": [0.1, 0.2, 0.3],
    })

    # Mock encode_sequences & build_model
    monkeypatch.setattr("ml_models.encoding.encode_sequences", lambda seqs, **kw: [[1]*20]*len(seqs))
    class DummyModel:
        def fit(self, X, y): pass
        def predict(self, X): return [0.1]*len(X)
    monkeypatch.setattr("ml_models.models.build_model", lambda t, m: DummyModel())

    res = run_single("ridge", "aac", 3, df, "regression", 42, 0.2, "none")
    assert isinstance(res, dict)
    assert set(res.keys()) >= {"model", "encoding", "rho", "p"}
    assert res["status"] == "ok"

from ml_models.models import build_model
import numpy as np

def test_classification_models():
    for name in ["logreg", "rf", "mlp"]:
        m = build_model("classification", name, seed=1)
        m.fit(np.random.randn(5, 10), np.array([0,1,0,1,0]))
        preds = m.predict(np.random.randn(2, 10))
        assert len(preds) == 2

def test_regression_models():
    for name in ["ridge", "rf", "mlp"]:
        m = build_model("regression", name, seed=1)
        m.fit(np.random.randn(5, 10), np.random.rand(5))
        preds = m.predict(np.random.randn(2, 10))
        assert preds.shape == (2,)

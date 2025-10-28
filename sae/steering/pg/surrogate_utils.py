import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from ml_models.encoding import encode_sequences

def load_or_train_surrogate(ridge_path, train_x, train_y, encoding: str):
    if os.path.exists(ridge_path):
        print(f"[INFO] Loading pretrained surrogate from {ridge_path}")
        return joblib.load(ridge_path)
    print(f"[INFO] Training new Ridge surrogate using encoding '{encoding}'")
    X = encode_sequences(train_x, encoding=encoding)
    y = np.array(train_y)
    ridge = Ridge(alpha=1.0).fit(X, y)
    joblib.dump(ridge, ridge_path)
    print(f"[INFO] Saved surrogate to {ridge_path}")
    return ridge

def predict_sequences(ridge, sequences, encoding: str):
    X = encode_sequences(sequences, encoding=encoding)
    return ridge.predict(X)

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

def predict_sequences(ridge, sequences, encoding: str, encoding_cache=None):
    if encoding_cache is None:
        X = encode_sequences(sequences, encoding=encoding)
    else:
        feats = []
        missing = []
        for s in sequences:
            cached = encoding_cache.get(s)
            if cached is None:
                missing.append(s)
            else:
                feats.append(cached)
        if missing:
            Xmiss = encode_sequences(missing, encoding=encoding)
            for s, Xi in zip(missing, Xmiss):
                import numpy as np
                Xi = Xi.reshape(1, -1) if Xi.ndim == 1 else Xi
                encoding_cache.set(s, Xi)
        feats = [encoding_cache.get(s) for s in sequences]
        import numpy as np
        X = np.vstack([f if f.ndim==2 else f.reshape(1,-1) for f in feats])
    return ridge.predict(X)

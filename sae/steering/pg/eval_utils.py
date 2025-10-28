import numpy as np
from .surrogate_utils import predict_sequences

def _spearman_np(y_true, y_pred):
    def rankdata(a):
        temp = a.argsort(kind='mergesort')
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(a))
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, ranks)
        avg = sums / counts
        return avg[inv]
    rt = rankdata(np.asarray(y_true))
    rp = rankdata(np.asarray(y_pred))
    rt = (rt - rt.mean()) / (rt.std() + 1e-12)
    rp = (rp - rp.mean()) / (rp.std() + 1e-12)
    return float((rt * rp).mean())

def eval_spearman(ridge, encoding, train_x, train_y, test_x, test_y):
    yhat_train = predict_sequences(ridge, list(train_x), encoding)
    yhat_test  = predict_sequences(ridge, list(test_x),  encoding)
    rho_train = _spearman_np(train_y, yhat_train)
    rho_test  = _spearman_np(test_y,  yhat_test)
    return rho_train, rho_test

def log_eval(epoch, ridge, encoding, train_x, train_y, test_x, test_y):
    rho_tr, rho_te = eval_spearman(ridge, encoding, train_x, train_y, test_x, test_y)
    print(f"[eval] epoch {epoch} spearman_train {rho_tr:.4f} spearman_test {rho_te:.4f}")
    return rho_tr, rho_te

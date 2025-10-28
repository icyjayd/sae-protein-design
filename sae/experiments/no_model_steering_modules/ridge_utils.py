import numpy as np, pandas as pd
from scipy.stats import spearmanr
import joblib
from ml_models.encoding import encode_sequences


AA20="ACDEFGHIKLMNPQRSTVWY"

from scipy.stats import spearmanr
import numpy as np
import pandas as pd

def safe_spearman(x, y):
    """
    Compute Spearman correlation and p-value safely, without omitting NaNs.
    If any element in x or y is NaN or computation fails, returns (NaN, NaN).
    """
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # Explicitly fail if any NaN in either array
        if np.isnan(x).any() or np.isnan(y).any():
            return np.nan, np.nan
        if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
            return np.nan, np.nan
        # Run Spearman directly
        r, p = spearmanr(x, y, nan_policy="propagate")
        if isinstance(r, (float, np.floating)):
            rho, pval = r, p
        else:
            rho, pval = r.correlation, r.pvalue
        return rho, pval
    except Exception as e:
        print(f"[WARN] Spearman failed: {e}")
        return np.nan, np.nan


def summarize(df, ranking_df=None):
    """
    Summarize results by split, returning both rho and p-values.
    If any correlation can't be computed, returns NaN for both.
    """
    rows = []
    for split, g in df.groupby("split"):
        rho1, p1 = safe_spearman(g["y_recon"], g["y_steer"])
        rho2, p2 = safe_spearman(np.abs(g["y_recon"]), np.abs(g["delta"]))
        row = {
            "split": split,
            "spearman_yrecon_vs_ysteer": rho1,
            "pval_yrecon_vs_ysteer": p1,
            "spearman_abs_yrecon_vs_abs_delta": rho2,
            "pval_abs_yrecon_vs_abs_delta": p2,
            "n_sequences": len(g),
            "n_nochange": int(g["no_change"].sum())
        }
        rows.append(row)
    df_sum = pd.DataFrame(rows)

    # merge with ranking data if provided
    if ranking_df is not None:
        if "latent" in df_sum.columns:
            df_sum = df_sum.rename(columns={"latent": "latent_index"})
        if "latent_index" in df_sum.columns and "latent_index" in ranking_df.columns:
            df_sum = df_sum.merge(
                ranking_df[["latent_index", "abs_corr"]],
                on="latent_index",
                how="left"
            )
    return df_sum


def encode_onehot(seq):
    v=np.zeros((20,),float)
    for c in seq:
        i=AA20.find(c)
        if i>=0:v[i]+=1
    return v

def encode_batch(seqs): return np.stack([encode_onehot(s) for s in seqs])

def score_with_ridge(ridge_path, seq_recon, seq_steer, encoding="onehot"):
    ridge=joblib.load(ridge_path)
    Xr = encode_sequences(seq_recon, encoding=encoding, use_cache=False)
    Xs = encode_sequences(seq_steer, encoding=encoding, use_cache=False)
    y_r=ridge.predict(Xr); y_s=ridge.predict(Xs)
    return y_r,y_s

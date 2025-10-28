import numpy as np, pandas as pd
from scipy.stats import spearmanr
import joblib

AA20="ACDEFGHIKLMNPQRSTVWY"

def safe_spearman(x,y):
    if len(x)<3 or np.allclose(x,x[0]) or np.allclose(y,y[0]): return np.nan
    return spearmanr(x,y).correlation

def encode_onehot(seq):
    v=np.zeros((20,),float)
    for c in seq:
        i=AA20.find(c)
        if i>=0:v[i]+=1
    return v

def encode_batch(seqs): return np.stack([encode_onehot(s) for s in seqs])

def score_with_ridge(ridge_path, seq_recon, seq_steer, encoding="onehot"):
    ridge=joblib.load(ridge_path)
    if encoding=="onehot":
        Xr=encode_batch(seq_recon); Xs=encode_batch(seq_steer)
    else:
        raise ValueError("Only onehot encoding currently implemented.")
    y_r=ridge.predict(Xr); y_s=ridge.predict(Xs)
    return y_r,y_s

def summarize(df, ranking_df=None):
    rows=[]
    for split,g in df.groupby("split"):
        row={
            "split":split,
            "spearman_yrecon_vs_ysteer":safe_spearman(g["y_recon"],g["y_steer"]),
            "spearman_abs_yrecon_vs_abs_delta":safe_spearman(np.abs(g["y_recon"]),np.abs(g["delta"])),
            "n_sequences":len(g),"n_nochange":int(g["no_change"].sum())
        }
        rows.append(row)
    df_sum=pd.DataFrame(rows)
    if ranking_df is not None and "latent" in df_sum.columns and "abs_corr" in ranking_df.columns:
        df_sum=df_sum.merge(ranking_df[["latent","abs_corr"]],on="latent",how="left")
    return df_sum

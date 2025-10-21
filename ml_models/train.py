import numpy as np,pandas as pd,joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from .data_utils import load_data
from .encoding import encode_aac,encode_kmer,encode_onehot,encode_esm
from .models import build_model
from .metrics import compute_metrics

def train_model(seq_file,labels_file=None,model_name="xgb",encoding="kmer",task=None,seed=42,outdir="runs/ml_model",**kwargs):
    Path(outdir).mkdir(parents=True,exist_ok=True)
    df=load_data(seq_file,labels_file)
    if task is None:
        try: pd.to_numeric(df["label"]); task="regression"
        except: task="classification"
    if encoding=="aac": X=encode_aac(df.sequence)
    elif encoding=="kmer": X=encode_kmer(df.sequence,k=kwargs.get("k",3))
    elif encoding=="onehot": X=encode_onehot(df.sequence,max_len=kwargs.get("max_len",512))
    elif encoding=="esm": X=encode_esm(df.sequence,device=kwargs.get("device","cpu"))
    else: raise ValueError("Bad encoding")
    pre=StandardScaler(with_mean=False); X=pre.fit_transform(X)
    y=df["label"]; le=None
    if task=="classification" and not np.issubdtype(y.dtype,np.number):
        le=LabelEncoder(); y=le.fit_transform(y)
    else: y=pd.to_numeric(y)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=seed,stratify=y if (task=="classification" and len(np.unique(y))>1) else None)
    m=build_model(task,model_name,seed); m.fit(Xtr,ytr)
    yp=m.predict(Xte); yp_prob=m.predict_proba(Xte) if hasattr(m,"predict_proba") else None
    met=compute_metrics(task,yte,yp,yp_prob)
    joblib.dump(m,f"{outdir}/model.pkl"); joblib.dump(pre,f"{outdir}/scaler.pkl")
    if le: joblib.dump(le,f"{outdir}/labels.pkl")
    return met

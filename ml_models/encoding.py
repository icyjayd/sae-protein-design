import numpy as np, torch
from itertools import product

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a:i for i,a in enumerate(AMINO_ACIDS)}

def encode_aac(seqs):
    X = np.zeros((len(seqs), len(AMINO_ACIDS)), dtype=np.float32)
    for i,s in enumerate(seqs):
        for ch in s:
            if ch in AA_TO_IDX: X[i,AA_TO_IDX[ch]]+=1
        X[i]/=max(1,len(s))
    return X

def encode_kmer(seqs,k=3):
    kmers = ["".join(p) for p in product(AMINO_ACIDS,repeat=k)]
    idx = {kmer:j for j,kmer in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)),dtype=np.float32)
    for i,s in enumerate(seqs):
        for t in range(len(s)-k+1):
            kmer=s[t:t+k]
            if kmer in idx: X[i,idx[kmer]]+=1
        X[i]/=max(1,len(s)-k+1)
    return X

def encode_onehot(seqs,max_len=512):
    X=np.zeros((len(seqs),max_len,len(AMINO_ACIDS)),dtype=np.float32)
    for i,s in enumerate(seqs):
        s=s[:max_len]
        for j,ch in enumerate(s):
            if ch in AA_TO_IDX: X[i,j,AA_TO_IDX[ch]]=1
    return X.reshape(len(seqs),-1)

def encode_esm(seqs,model_name="facebook/esm2_t6_8M_UR50D",device="cpu"):
    from transformers import EsmTokenizer,EsmModel
    tok=EsmTokenizer.from_pretrained(model_name)
    mdl=EsmModel.from_pretrained(model_name).eval().to(device)
    outs=[]
    with torch.no_grad():
        for s in seqs:
            t=tok(s,return_tensors="pt").to(device)
            rep=mdl(**t).last_hidden_state.mean(1)
            outs.append(rep.cpu().numpy().flatten())
    return np.stack(outs)

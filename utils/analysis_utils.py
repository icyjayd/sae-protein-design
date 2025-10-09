\"\"\"Analysis helpers: small utilities for plotting and metrics.\"\"\"
import numpy as np
from sklearn.metrics import pairwise_distances

def sequence_diversity_metric(seqs):
    # simple Hamming-like distance on equal-length sequences
    n = len(seqs)
    if n == 0:
        return np.array([])
    L = len(seqs[0])
    A = len(seqs)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            D[i,j] = sum(1 for a,b in zip(seqs[i], seqs[j]) if a!=b) / L
            D[j,i] = D[i,j]
    return D

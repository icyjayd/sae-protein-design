\"\"\"Generate synthetic sequences and simple encodings (same as prior skeleton).\"\"\"
import numpy as np
def generate_synthetic_dataset(n, L, alphabet, motif=None):
    rng = np.random.RandomState(0)
    alphabet = list(alphabet)
    seqs = []
    ids = []
    for i in range(n):
        seq = ''.join(rng.choice(alphabet, size=L))
        if motif and rng.rand() < 0.3:
            pos = rng.randint(0, L - len(motif) + 1)
            seq = seq[:pos] + motif + seq[pos+len(motif):]
        seqs.append(seq)
        ids.append(f"seq_{i}")
    return seqs, ids

def one_hot_encode(seqs, alphabet):
    A = len(alphabet)
    L = len(seqs[0])
    idx = {a:i for i,a in enumerate(alphabet)}
    X = np.zeros((len(seqs), L, A), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            X[i, j, idx[ch]] = 1.0
    return X.reshape(len(seqs), L*A)


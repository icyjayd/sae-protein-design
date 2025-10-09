\"\"\"Data utilities: synthetic data generator and simple IO.\"\"\"
import numpy as np
from pathlib import Path

def generate_synthetic_dataset(n, L, alphabet, motif=None):
    \"\"\"Generate n random sequences length L from alphabet. Optionally seed with a motif.\"\"\"
    rng = np.random.RandomState(0)
    alphabet = list(alphabet)
    seqs = []
    ids = []
    for i in range(n):
        seq = ''.join(rng.choice(alphabet, size=L))
        # inject motif occasionally
        if motif and rng.rand() < 0.3:
            pos = rng.randint(0, L - len(motif) + 1)
            seq = seq[:pos] + motif + seq[pos+len(motif):]
        seqs.append(seq)
        ids.append(f"seq_{i}")
    return seqs, ids

def one_hot_encode(seqs, alphabet):
    \"\"\"Return array (n, L, A) flattened to (n, L*A).\"\"\"
    A = len(alphabet)
    L = len(seqs[0])
    idx = {a:i for i,a in enumerate(alphabet)}
    X = np.zeros((len(seqs), L, A), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            X[i, j, idx[ch]] = 1.0
    return X.reshape(len(seqs), L*A)

def load_saved_onehot(seqs_np_path, alphabet):
    seqs = np.load(seqs_np_path, allow_pickle=True)
    return one_hot_encode(seqs.tolist(), alphabet)

def sequences_from_onehot(onehot_flat, alphabet):
    \"\"\"Convert flattened one-hot (L*A) to sequence string.\"\"\"
    n, LA = onehot_flat.shape
    A = len(alphabet)
    L = LA // A
    seqs = []
    for i in range(n):
        arr = onehot_flat[i].reshape(L, A)
        idxs = arr.argmax(axis=1)
        seqs.append(''.join(alphabet[k] for k in idxs))
    return seqs

def save_fasta(seqs, ids, outpath):
    with open(outpath, 'w') as f:
        for sid, s in zip(ids, seqs):
            f.write(f">{sid}\n{s}\n")

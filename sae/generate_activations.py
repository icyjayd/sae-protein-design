
#!/usr/bin/env python3
import numpy as np
from pathlib import Path
OUT = Path('outputs'); OUT.mkdir(exist_ok=True)
rng = np.random.RandomState(0)
N, D = 300, 256
A = rng.normal(0,1,(N,D))
seqs = ['M' + ''.join(rng.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=99)) for _ in range(N)]
y = rng.normal(0,1,N)
np.save(OUT/'activations.npy', A)
np.save(OUT/'sequences.npy', np.array(seqs, dtype=object))
np.save(OUT/'labels.npy', y)
print('Wrote toy activations, sequences, labels to outputs/')

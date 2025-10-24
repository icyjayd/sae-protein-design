# sae/decoder/data.py
import torch
from torch.utils.data import Dataset, DataLoader
from sae.utils.esm_utils import encode_sequence
from typing import List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}


# =========================================================
# 1. Latent Sequence Dataset
# =========================================================
class LatentSequenceDataset(Dataset):
    """
    Builds (latent, token_sequence) pairs for training the sequence decoder.
    """

    def __init__(self, sequences: List[str], sae_model, esm_model, tokenizer, device="cpu"):
        self.device = device
        self.samples = []

        for seq in sequences:
            token_reps, _ = encode_sequence(seq, esm_model, tokenizer, device=device)
            latent = sae_model.encode(token_reps[1:-1]).mean(0).detach().cpu()
            token_indices = torch.tensor([AA_TO_IDX.get(aa, 0) for aa in seq], dtype=torch.long)
            self.samples.append((latent, token_indices))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_dataloader(dataset, batch_size=32, shuffle=True):
    def collate(batch):
        latents, tokens = zip(*batch)
        latents = torch.stack(latents)
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=-100)
        return latents, tokens
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# =========================================================
# 2. Sequence Loading, Splitting, and Caching
# =========================================================
CACHE_ROOT = Path("decoder_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def get_cache_dir(experiment: str) -> Path:
    d = CACHE_ROOT / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_or_create_splits(input_path: str, experiment: str) -> Tuple[list, list, bool]:
    """
    Loads sequences and (optionally) splits from CSV, NPY, or TXT.
    If no cached split exists for the experiment, creates one and saves it.
    Returns: (train_sequences, test_sequences, used_cache)
    """
    path = Path(input_path)
    cache_dir = get_cache_dir(experiment)
    cache_train = cache_dir / "train_sequences.txt"
    cache_test = cache_dir / "test_sequences.txt"

    # --- Cached split exists ---
    if cache_train.exists() and cache_test.exists():
        print(f"[INFO] Using cached split for experiment '{experiment}'.")
        train = [l.strip() for l in cache_train.read_text().splitlines() if l.strip()]
        test = [l.strip() for l in cache_test.read_text().splitlines() if l.strip()]
        return train, test, True

    # --- Load fresh data ---
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "sequence" not in df.columns:
            raise ValueError("CSV must contain a 'sequence' column.")
        if "split" in df.columns:
            train = df[df["split"].str.lower() == "train"]["sequence"].dropna().tolist()
            test = df[df["split"].str.lower() == "test"]["sequence"].dropna().tolist()
        else:
            seqs = df["sequence"].dropna().tolist()
            train, test = train_test_split(seqs, test_size=0.2, random_state=42)
            print(f"[INFO] Created random 80/20 split from CSV.")
    elif path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        seqs = arr.tolist() if arr.dtype == object else [str(s) for s in arr]
        train, test = train_test_split(seqs, test_size=0.2, random_state=42)
        print(f"[INFO] Created random 80/20 split from NPY.")
    else:
        with open(path) as f:
            seqs = [line.strip() for line in f if line.strip()]
        train, test = train_test_split(seqs, test_size=0.2, random_state=42)
        print(f"[INFO] Created random 80/20 split from TXT.")

    # --- Cache the new split ---
    cache_train.write_text("\n".join(train))
    cache_test.write_text("\n".join(test))
    print(f"[INFO] Saved new split to {cache_dir}")
    return train, test, False

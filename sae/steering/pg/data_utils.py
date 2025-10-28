from turtle import pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
class GeneralDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i], self.labels[i]

def sample_train_test(df, sample_size=10):
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    train_df = train_df.sample(sample_size, random_state=42)
    test_df = test_df.sample(sample_size, random_state=42)
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    return df
def load_data(seq_path, label_path=None):
    if seq_path.endswith(".csv"):
        df = pd.read_csv(seq_path)
        if label_path is None and "label" in df.columns:
            num_samples = 16
            print(f"sampling {num_samples} seqs for testing")
            df = sample_train_test(df, sample_size=num_samples)
            seqs = df["sequence"].tolist()
            labels = df["label"].to_numpy()
        elif label_path is not None:
            seqs = df["sequence"].tolist()
            labels = np.load(label_path)
        else:
            raise ValueError("Need either 'label' column in CSV or external label file (.npy).")
        splits = df["split"].tolist() if "split" in df.columns else None
    elif seq_path.endswith(".npy") and label_path is not None:
        seqs = np.load(seq_path, allow_pickle=True)
        labels = np.load(label_path)
        splits = None
    else:
        raise ValueError("Unsupported file combination.")
    return seqs, labels, splits

def split_dataset(seqs, labels, splits):
    if splits is None:
        n = len(seqs); idx = int(0.9 * n)
        return (seqs[:idx], labels[:idx]), (seqs[idx:], labels[idx:])
    train_idx = [i for i,s in enumerate(splits) if str(s).lower() == "train"]
    test_idx  = [i for i,s in enumerate(splits) if str(s).lower() == "test"]
    seqs, labels = np.array(seqs), np.array(labels)
    return (seqs[train_idx], labels[train_idx]), (seqs[test_idx], labels[test_idx])

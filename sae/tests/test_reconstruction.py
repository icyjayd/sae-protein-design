import pytest
import torch
import numpy as np
import sys
from pathlib import Path
import os
from transformers import EsmForMaskedLM, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(1, os.path.join(REPO_ROOT, 'interplm'))
from interplm.sae.inference import load_sae_from_hf

from utils.esm_utils import encode_sequence
from utils.grade_reconstructions import grade_pair, mean_grade
import warnings
# Dummy SAE projection weights for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_dict = {
    "8m": {"hf_model_name": "facebook/esm2_t6_8M_UR50D", "sae_model_name": "esm2-8m", "layer": 6},
    "650m": {"hf_model_name": "facebook/esm2_t33_650M_UR50D", "sae_model_name": "esm2-650m", "layer": 33},
}

LATENT_DIM = 64  # latent size for projection
MODEL = "650m"  # "8m" or "650m"

def load_interplm_sae(model_name="esm2-8m", layer=4, device="cpu"):
    """
    Loads a pretrained InterPLM SAE from Hugging Face for ESM2 models.
    """
    print(f"[INFO] Loading InterPLM SAE from {model_name}, layer {layer}")
    sae = load_sae_from_hf(model_name, plm_layer=layer)
    sae.to(device)
    sae.eval()
    return sae

def check_seq(sequence, model, tokenizer, sae, device="cpu"):
    token_reps, _ = encode_sequence(sequence, model, tokenizer, device=device)
    token_reps = token_reps.unsqueeze(0)           # (1, L, hidden_dim)
    L = token_reps.shape[1] 
    # Flatten to feed per-token vectors through SAE
    reconstructed_tokens = []
    for i in range(1, L-1):  # iterate over L
        token_vec = token_reps[0, i].unsqueeze(0)  # (1, hidden_dim)
        latent = sae.encode(token_vec)
        recon = sae.decode(latent).squeeze()
        reconstructed_tokens.append(recon)
    reconstructed = torch.stack(reconstructed_tokens).detach()  # (1, L, hidden_dim)
    with torch.no_grad():
        logits = model.lm_head(reconstructed)     # (L, vocab_size)
        # print(f"logits shape: {logits.shape}")
        predicted_ids = torch.argmax(logits, dim=-1)
        print(f"predicted_ids shape: {predicted_ids.shape}")
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        decoded_seq = "".join([tok for tok in predicted_tokens])# if tok not in {"<cls>", "<eos>", "<pad>", "<mask>"}
    
    print((f"\nseq: {sequence}\n"
                   f"dec: {decoded_seq}\n"
                   f"{sequence == decoded_seq}"))
    
    return decoded_seq, sequence == decoded_seq

def get_models():
    hf_model_name = esm_dict[MODEL]["hf_model_name"]
    layer = esm_dict[MODEL]["layer"]
    sae_model_name = esm_dict[MODEL]["sae_model_name"]
    model = EsmForMaskedLM.from_pretrained(hf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model.to(device)
    model.eval()
    sae = load_interplm_sae(model_name=sae_model_name, layer=layer, device=device)
    sae.eval()
    return model, tokenizer, sae

def test_sequence_reconstruction():
    model, tokenizer, sae = get_models()
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLQK"
    decoded_seq, success = check_seq(sequence, model, tokenizer, sae, device=device)
    score = grade_pair(sequence, decoded_seq)['final_score']
    # assert success, f"Reconstruction failed: {decoded_seq} != {sequence}"
    assert score > 0.95, f"Reconstruction score is too low: {score:.4f}"
def test_multiple_sequences():
    model, tokenizer, sae = get_models()
    sequences = []
    min_len = np.inf
    max_len = -np.inf
    with open('tests/random_genes.txt', 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq)
                min_len = min(min_len, len(seq))
                max_len = max(max_len, len(seq))
    successes = []
    decodes = []
    scores = []
    print(f"Testing {len(sequences)} sequences, length range: {min_len}-{max_len}")
    for i, seq in enumerate(sequences):
        print(f"\n--- Testing sequence {i+1}/{len(sequences)} ---")
        decoded_seq, success = check_seq(seq, model, tokenizer, sae, device=device)
        successes.append(success)
        decodes.append(decoded_seq)
        scores.append(grade_pair(seq, decoded_seq)['final_score'])
    print(("=====Score summary=====\n"
           f"Min: {min(scores):.4f}\n"
           f"Max: {max(scores):.4f}\n"
           f"Mean: {sum(scores) / len(scores):.4f}\n"
           f"Std: {np.std(scores):.4f}\n"
           ))
    mean_score = mean_grade(list(zip(sequences, decodes)), csv_path=f"tests/reconstruction_report_{MODEL}.csv") 
    print(f"\nSummary: {sum(successes)}/{len(successes)} ({sum(successes) / len(successes) * 100:.2f}%) sequences reconstructed successfully.")
    assert mean_score > 0.95, f"Mean reconstruction score is too low: {mean_score:.4f}"
    # for i in range(len(successes)):
    #     seq = sequences[i]
    #     decoded_seq = decodes[i]
    #     success = successes[i]
    #     assert success, f"Reconstruction failed: {decoded_seq} != {seq}"



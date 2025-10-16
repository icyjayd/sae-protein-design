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

from utils.esm_utils import load_esm2_model, encode_sequence
import warnings
# Dummy SAE projection weights for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 64  # latent size for projection

@pytest.fixture(scope="module")
def esm_model():
    model, tokenizer = load_esm2_model("facebook/esm2_t6_8M_UR50D", device=device)
    return model, tokenizer

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
    warnings.warn((
        f"token_reps shape: {token_reps.shape} | "
        f"L: {L}"
        ))
    for i in range(1, L-1):  # iterate over L
        token_vec = token_reps[0, i].unsqueeze(0)  # (1, hidden_dim)
        latent = sae.encode(token_vec)
        recon = sae.decode(latent).squeeze()
        reconstructed_tokens.append(recon)
    # warnings.warn((
    #     f"Number of reconstructed tokens: {len(reconstructed_tokens)} |"
    #     f"token_vec shape: {token_vec.shape} |"
    #     f"latent shape: {latent.shape} |"
    #     f"recon shape: {recon.shape} |"
    #     f"len reconstructed_tokens: {len(reconstructed_tokens)} |"
    #     f"reconstructed_tokens[0] shape: {reconstructed_tokens[0].shape}"
    #     ))
    reconstructed = torch.stack(reconstructed_tokens).detach()  # (1, L, hidden_dim)
    with torch.no_grad():
        # warnings.warn(f"latent shape: {latent.shape}")
        # warnings.warn(f"reconstructed shape: {reconstructed.shape}")
        logits = model.lm_head(reconstructed)     # (L, vocab_size)
        # warnings.warn(f"logits shape: {logits.shape}")
        predicted_ids = torch.argmax(logits, dim=-1)
        # warnings.warn(f"predicted_ids shape: {predicted_ids.shape}")
        # warnings.warn(f"len predicted_ids: {len(predicted_ids)} ")
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        decoded_seq = "".join(
            tok for tok in predicted_tokens if tok not in {"<cls>", "<eos>", "<pad>", "<mask>"}
    )
    assert len(decoded_seq) == len(sequence), f"Decoded sequence length mismatch: {len(decoded_seq)} != {len(sequence)}"
    # Compute correlation
    # corr = np.corrcoef(original_vector.flatten(), reconstructed.flatten())[0, 1]
    # print(f"Correlation between original and reconstructed: {corr:.4f}")
    # assert corr > 0.5, f"Reconstruction correlation too low: {corr:.4f}"
    warnings.warn((f"\nseq: {sequence}\n"
                   f"dec: {decoded_seq}"))
    assert decoded_seq == sequence, f"Reconstruction failed: {decoded_seq} !=\n {sequence}"
def get_models():
    model_name = "facebook/esm2_t6_8M_UR50D"
    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    sae = load_interplm_sae(model_name="esm2-8m", layer=6, device=device)
    sae.eval()
    return model, tokenizer, sae
def test_sequence_reconstruction():
    model, tokenizer, sae = get_models()
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLQK"
    check_seq(sequence, model, tokenizer, sae, device=device)
    # warnings.warn(f"sequence length: {len(sequence)}")
    # Step 1: Encode sequence → pooled embedding

           

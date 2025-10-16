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

from utils.esm_utils import load_esm2_model, encode_sequence, decode_activation
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

def test_sequence_reconstruction(esm_model):
    model_name = "facebook/esm2_t6_8M_UR50D"
    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLQK"

    # Step 1: Encode sequence → pooled embedding
    full_reps, pooled = encode_sequence(sequence, model, tokenizer, device=device)
    original_vector = torch.tensor(pooled.cpu().numpy().flatten(), dtype=torch.float32)

    token_reps = full_reps[0]  # (L, hidden_size)

    # Step 2: Create compatible projection matrices
    #  encoder:  hidden_size → latent_dim
    #  decoder:  latent_dim → hidden_size
    sae = load_interplm_sae(model_name="esm2-8m", layer=4, device=device)

    # Encode and decode with SAE
    acts = torch.tensor(original_vector, dtype=torch.float32).unsqueeze(0)
    # latent = sae.encoder(acts)
    # reconstructed = sae.decoder(latent).detach().cpu().numpy().flatten()
    latent = sae.encode(token_reps)
    reconstructed = sae.decode(latent).detach()
    with torch.no_grad():
        logits = model.lm_head(reconstructed)
        predicted_ids = torch.argmax(logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
    decoded_seq = "".join([t for t in predicted_tokens if t not in ("<pad>", "<cls>", "</s>")])

    # Compute correlation
    # corr = np.corrcoef(original_vector.flatten(), reconstructed.flatten())[0, 1]
    # print(f"Correlation between original and reconstructed: {corr:.4f}")
    # assert corr > 0.5, f"Reconstruction correlation too low: {corr:.4f}"
    assert decoded_seq == sequence, f"Reconstruction failed: {decoded_seq} != {sequence}"

    # warnings.warn((f"[PASS] Sequence reconstruction test passed with correlation {corr:.4f} |"
           
    #        ))
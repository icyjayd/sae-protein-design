"""
Integration test using real pretrained ESM2 + InterPLM SAE models
to verify whole-sequence reconstruction via perturb_and_decode_whole.
"""

import pytest
import torch
from transformers import EsmForMaskedLM, EsmTokenizer
from mod_man_utils import add_module
add_module("interplm")
from interplm.sae.inference import load_sae_from_hf
from sae.utils.esm_utils import perturb_and_decode_whole, encode_whole_sequence


@pytest.mark.integration
@pytest.mark.slow
def test_real_model_reconstruction():
    """
    Loads real pretrained ESM2 + InterPLM SAE models and checks that
    perturb_and_decode_whole reconstructs a valid amino acid sequence.
    Always runs, even if no GPU is available.
    """
    # --- Load models (always run; uses CPU fallback) ---
    model_name = "facebook/esm2_t6_8M_UR50D"
    sae_model = load_sae_from_hf("esm2-8m", plm_layer=6)
    esm_model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model.to(device).eval()
    sae_model.to(device).eval()

    # --- Test sequence ---
    sequence = "MKTAYIAKQRQISFVKSHFSRQDILD"

    # --- Encode pooled vector ---
    token_reps, pooled = encode_whole_sequence(sequence, esm_model, tokenizer, device=device)
    assert token_reps.ndim == 2
    assert pooled.ndim == 1
    assert pooled.shape[0] > 0

    # --- Run reconstruction with a small perturbation ---
    decoded_seq = perturb_and_decode_whole(
        sequence=sequence,
        sae_model=sae_model,
        esm_model=esm_model,
        tokenizer=tokenizer,
        latent_deltas={0: 0.2},  # small global perturbation
        device=str(device)
    )

    # --- Print and validate results ---
    print("\n[INFO] Original:", sequence)
    print("[INFO] Reconstructed:", decoded_seq)

    assert isinstance(decoded_seq, str)
    assert len(decoded_seq) > 0
    assert all(c.isalpha() for c in decoded_seq), f"Decoded sequence invalid: {decoded_seq}"
    assert decoded_seq[0] in "ACDEFGHIKLMNPQRSTVWY", "Output not valid amino acids"

    # --- Similarity check (loose threshold) ---
    similarity = sum(a == b for a, b in zip(sequence, decoded_seq)) / len(sequence)
    print(f"[INFO] Reconstruction similarity: {similarity:.2f}")
    assert similarity > 0.3

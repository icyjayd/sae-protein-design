"""
ESM Utilities for Per-Token SAE Perturbation.

This file provides the core functions for:
1. Loading the ESM model (with LM head) and tokenizer.
2. Encoding a sequence into per-token representations.
3. Running the full "Encode -> Perturb -> Decode" pipeline
   on a per-token basis.
"""

import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from typing import Dict, Optional, List

# --- 1. Model Loading ---

def load_esm2_model(model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = "cpu"):
    """
    Loads the ESM model *with LM head* and tokenizer.
    The LM head is essential for decoding token embeddings back to logits.
    """
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

# --- 2. Encoding ---

def encode_sequence(sequence: str, model: EsmForMaskedLM, tokenizer: EsmTokenizer, device: str = "cpu"):
    """
    Encodes a sequence into per-token representations.
    
    Returns:
        token_reps: (L, D) tensor of token representations (no batch dim).
        pooled: (D,) tensor of mean-pooled representation.
    """
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # We use the *last* hidden state, which is what the LM head sees
        token_reps = outputs.hidden_states[-1].squeeze(0)
    
    # Pooled representation (ignoring special tokens)
    pooled = token_reps[1:-1].mean(dim=0)
    return token_reps, pooled

# --- 3. Perturbation and Decoding ---

# TODO: Implement position-wise (surgical) perturbation.
# The current `perturbations` dict applies a global delta to the
# specified latent dim across *all* tokens in the sequence.
# A more advanced version would allow specifying *which*
# tokens (i.e., sequence positions) to perturb, e.g.,
# by also passing a `target_positions: List[int]` argument.
# This would allow for more targeted, intelligent design based on
# per-token latent activations.

def perturb_and_decode(
    sequence: str,
    sae_model: torch.nn.Module,
    esm_model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    perturbations: Optional[Dict[int, float]] = None,
    device: str = "cpu"
) -> str:
    """
    Performs a full per-token encode -> perturb -> decode pipeline.
    
    This function is a stateless version of the logic found in
    the RealSAE.perturb_and_decode method.

    Args:
        sequence: The input protein sequence string.
        sae_model: The trained SAE model.
        esm_model: The ESM2 model (must have LM head).
        tokenizer: The ESM2 tokenizer.
        perturbations: A dictionary mapping latent_index -> delta_strength.
                       e.g., {4092: 10.0, 1024: -5.0}
                       If None or empty, performs reconstruction.
        device: The device to run on ('cuda' or 'cpu').

    Returns:
        The decoded protein sequence string.
    """
    if perturbations is None:
        perturbations = {}

    # 1. ENCODE SEQUENCE (Per-Token)
    # token_reps is (L, D), where L = len(sequence) + 2 (for <cls> and <eos>)
    token_reps, _ = encode_sequence(sequence, esm_model, tokenizer, device=device)
    L = token_reps.shape[0]

    reconstructed_token_embeddings = []
    
    # We iterate over the *real* tokens, skipping <cls> and <eos>
    for i in range(1, L - 1):
        token_vec = token_reps[i].unsqueeze(0)  # (1, D)
        
        # 2. ENCODE TOKEN (SAE)
        # Pass token embedding through SAE
        latent = sae_model.encode(token_vec) # (1, d_sae)
        # 3. PERTURB LATENT
        if perturbations:
            for dim, delta in perturbations.items():
                if dim < latent.shape[-1]:
                    latent[0, dim] += delta

        # 4. DECODE TOKEN (SAE)
        # Reconstruct token embedding from (potentially perturbed) latent
        recon_token_embedding = sae_model.decode(latent) # (1, D)
        reconstructed_token_embeddings.append(recon_token_embedding.squeeze(0))

    # We now have a list of (L-2) reconstructed token embeddings
    reconstructed_embeddings = torch.stack(reconstructed_token_embeddings).detach()  # (L-2, D)

    # 5. DECODE SEQUENCE (ESM LM Head)
    with torch.no_grad():
        # Pass the (L-2, D) tensor through the LM head to get logits
        logits = esm_model.lm_head(reconstructed_embeddings) # (L-2, vocab_size)
        
        # Get the most likely token ID for each position
        predicted_ids = torch.argmax(logits, dim=-1) # (L-2,)
        
        # Convert token IDs back to token strings
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        
        # Join tokens into a final sequence, filtering out special tokens
        # (though they shouldn't be here since we skipped <cls> and <eos>)
        decoded_seq = "".join(
            [tok for tok in predicted_tokens if tok not in {"<cls>", "<eos>", "<pad>", "<mask>"}]
        )

    return decoded_seq


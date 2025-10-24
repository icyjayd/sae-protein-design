"""
ESM Utilities for Per-Token SAE Perturbation.

This file provides the core functions for:
1. Loading the ESM model (with LM head) and tokenizer.
2. Encoding a sequence into per-token representations.
3. Running the full "Encode -> Perturb -> Decode" pipeline
   on a per-token basis.
4. Getting the per-token activation matrix.
"""

import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from typing import Dict, Optional, List, Tuple # Added Tuple

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
                    Includes <cls> and <eos> tokens.
        pooled: (D,) tensor of mean-pooled representation (real tokens only).
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

def perturb_and_decode(
    sequence: str,
    sae_model: torch.nn.Module,
    esm_model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    # --- MODIFIED ARGUMENT ---
    # This new structure allows specifying *which* latents to perturb
    # at *which* specific positions.
    # Example: { 10: {4092: 10.0}, 22: {4092: 12.0, 800: -5.0} }
    surgical_perturbations: Optional[Dict[int, Dict[int, float]]] = None,
    device: str = "cpu"
) -> str:
    """
    Performs a full per-token encode -> perturb -> decode pipeline.
    
    Args:
        sequence: The input protein sequence string.
        sae_model: The trained SAE model.
        esm_model: The ESM2 model (must have LM head).
        tokenizer: The ESM2 tokenizer.
        surgical_perturbations: (NEW) A dictionary mapping a 
            token_position_index -> { latent_index -> delta_strength }.
            Positions are 1-based (relative to the start of the 
            full token tensor, so 1 is the first real token).
        device: The device to run on ('cuda' or 'cpu').

    Returns:
        The decoded protein sequence string.
    """
    if surgical_perturbations is None:
        surgical_perturbations = {}

    # 1. ENCODE SEQUENCE (Per-Token)
    # token_reps is (L, D), where L = len(sequence) + 2 (for <cls> and <eos>)
    token_reps, _ = encode_sequence(sequence, esm_model, tokenizer, device=device)
    L = token_reps.shape[0]

    reconstructed_token_embeddings = []
    
    # We iterate over the *real* tokens, skipping <cls> (idx 0) and <eos> (idx L-1)
    for i in range(1, L - 1):
        token_vec = token_reps[i].unsqueeze(0)  # (1, D)
        
        # 2. ENCODE TOKEN (SAE)
        latent = sae_model.encode(token_vec) # (1, d_sae)

        # 3. PERTURB LATENT (SURGICAL)
        # --- MODIFIED LOGIC ---
        # Check if the *current token index 'i'* is a key in our map
        if i in surgical_perturbations:
            # Get the inner dict of {latent: delta} pairs for this position
            latent_deltas = surgical_perturbations[i]
            for dim, delta in latent_deltas.items():
                if dim < latent.shape[-1]:
                    latent[0, dim] += delta
        # --- END MODIFIED LOGIC ---

        # 4. DECODE TOKEN (SAE)
        recon_token_embedding = sae_model.decode(latent) # (1, D)
        reconstructed_token_embeddings.append(recon_token_embedding.squeeze(0))

    reconstructed_embeddings = torch.stack(reconstructed_token_embeddings).detach()  # (L-2, D)

    # 5. DECODE SEQUENCE (ESM LM Head)
    with torch.no_grad():
        logits = esm_model.lm_head(reconstructed_embeddings) # (L-2, vocab_size)
        predicted_ids = torch.argmax(logits, dim=-1) # (L-2,)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        
        decoded_seq = "".join(
            [tok for tok in predicted_tokens if tok not in {"<cls>", "<eos>", "<pad>", "<mask>"}]
        )

    return decoded_seq

# --- 4. Activation Matrix Utility ---

def get_activation_matrix(
    sequence: str,
    sae_model: torch.nn.Module,
    esm_model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Gets the per-token SAE latent activations for a sequence.
    
    Args:
        sequence: The input protein sequence string.
        sae_model: The trained SAE model.
        esm_model: The ESM2 model (must have LM head).
        tokenizer: The ESM2 tokenizer.
        device: The device to run on ('cuda' or 'cpu').
        
    Returns:
        A (L-2, 10240) tensor, where L is the sequence length + 2.
        Each row is the latent activation vector for a single amino acid.
    """
    # 1. Get per-token ESM embeddings
    # token_reps is (L, D), including <cls> and <eos>
    token_reps, _ = encode_sequence(sequence, esm_model, tokenizer, device=device)
    L = token_reps.shape[0]
    
    # 2. Get real token embeddings (skip <cls> and <eos>)
    # real_token_reps is (L-2, D)
    real_token_reps = token_reps[1:-1] 
    
    # 3. Pass all token embeddings through the SAE encoder at once
    with torch.no_grad():
        activations = sae_model.encode(real_token_reps) # (L-2, 10240)
        
    return activations


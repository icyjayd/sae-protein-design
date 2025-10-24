"""
ESM Utilities for Per-Token and Whole-Sequence SAE Perturbation.

This file provides the core functions for:
1. Loading the ESM model (with LM head) and tokenizer.
2. Encoding a sequence into per-token representations.
3. Running the full "Encode -> Perturb -> Decode" pipeline
   on a per-token basis.
4. Getting the per-token activation matrix.
5. (NEW) Whole-sequence perturbation and decoding
   using the mean activation vector of the sequence.
"""

import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from typing import Dict, Optional, List, Tuple

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
        token_reps = outputs.hidden_states[-1].squeeze(0)
    
    # Pooled representation (ignoring special tokens)
    pooled = token_reps[1:-1].mean(dim=0)
    return token_reps, pooled


# --- 3. Perturbation and Decoding (Per-Token) ---

def perturb_and_decode(
    sequence: str,
    sae_model: torch.nn.Module,
    esm_model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
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
        surgical_perturbations: A dictionary mapping 
            token_position_index -> { latent_index -> delta_strength }.
            Positions are 1-based (so 1 is the first real token).
        device: The device to run on ('cuda' or 'cpu').

    Returns:
        The decoded protein sequence string.
    """
    if surgical_perturbations is None:
        surgical_perturbations = {}

    token_reps, _ = encode_sequence(sequence, esm_model, tokenizer, device=device)
    L = token_reps.shape[0]
    reconstructed_token_embeddings = []

    for i in range(1, L - 1):
        token_vec = token_reps[i].unsqueeze(0)
        latent = sae_model.encode(token_vec)

        if i in surgical_perturbations:
            latent_deltas = surgical_perturbations[i]
            for dim, delta in latent_deltas.items():
                if dim < latent.shape[-1]:
                    latent[0, dim] += delta

        recon_token_embedding = sae_model.decode(latent)
        reconstructed_token_embeddings.append(recon_token_embedding.squeeze(0))

    reconstructed_embeddings = torch.stack(reconstructed_token_embeddings).detach()

    with torch.no_grad():
        logits = esm_model.lm_head(reconstructed_embeddings)
        predicted_ids = torch.argmax(logits, dim=-1)
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
    """
    token_reps, _ = encode_sequence(sequence, esm_model, tokenizer, device=device)
    real_token_reps = token_reps[1:-1]
    with torch.no_grad():
        activations = sae_model.encode(real_token_reps)
    return activations


# =====================================================================
# --- 5. Whole-Sequence Perturbation and Decoding (NEW) ---
# =====================================================================

def encode_whole_sequence(sequence: str, model, tokenizer, device: str = "cpu", use_cls: bool = True):
    """
    Encodes the sequence and returns:
        token_reps: (L, D)
        pooled:     (D,) vector, either the <cls> token or mean-pooled real tokens
    """
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("TOKENS:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()))
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        token_reps = outputs.hidden_states[-1].squeeze(0)  # (L, D)
    if use_cls:
        pooled = token_reps[0]              # <cls> token embedding
    else:
        pooled = token_reps[1:-1].mean(0)   # mean over real residues
    
    return token_reps, pooled
    
def perturb_and_decode_whole(
    sequence: str,
    sae_model: torch.nn.Module,
    esm_model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    latent_deltas: Optional[Dict[int, float]] = None,
    device: str = "cpu"
) -> str:
    """
    Encode -> Perturb -> Decode using *the mean activation vector*.
    Allows testing true causal control over individual latents globally.

    Args:
        sequence: Protein sequence string.
        sae_model: Trained SAE model.
        esm_model: ESM2 model (with LM head).
        tokenizer: ESM2 tokenizer.
        latent_deltas: {latent_index: delta_value} for global perturbation.
        device: 'cpu' or 'cuda'.

    Returns:
        Reconstructed protein sequence string.
    """
    latent_deltas = latent_deltas or {}

    _, pooled = encode_whole_sequence(sequence, esm_model, tokenizer, device=device)

    with torch.no_grad():
        latent = sae_model.encode(pooled.unsqueeze(0))

        for dim, delta in latent_deltas.items():
            if dim < latent.shape[-1]:
                latent[0, dim] += delta

        recon_embedding = sae_model.decode(latent)  # (1, D)
        L = len(sequence)
        recon_matrix = recon_embedding.repeat(L, 1)  # (L, D)

        logits = esm_model.lm_head(recon_matrix)
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())

        decoded_seq = "".join(
            [tok for tok in predicted_tokens if tok not in {"<cls>", "<eos>", "<pad>", "<mask>"}]
        )

    return decoded_seq

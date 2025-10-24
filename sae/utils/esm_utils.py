# sae/utils/esm_utils.py

# --- FIX: Import EsmForMaskedLM, which includes the lm_head for decoding ---
from transformers import EsmForMaskedLM, EsmTokenizer
import torch

def load_esm2_model(model_name="facebook/esm2_t6_8M_UR50D", device="cpu"):
    """
    Loads the ESM model *for Masked LM* (which includes the lm_head)
    and tokenizer.
    """
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    # --- FIX: Load EsmForMaskedLM ---
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def encode_sequence(sequence, model, tokenizer, device="cpu"):
    """
    Returns embedding (L x D) and pooled representation (D,)
    Note: L includes <cls> and <eos> tokens.
    """
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # use the last hidden state (before LM head)
        # This is (batch, L, D), so we squeeze batch dim
        token_reps = outputs.hidden_states[-1].squeeze(0) 
    
    # Return token reps *and* mean-pooled reps
    # We pool over the sequence dim (0), ignoring <cls> and <eos>
    pooled = token_reps[1:-1].mean(dim=0) 
    return token_reps, pooled

def decode_activation(activation, projection_weights):
    """
    Reconstruct the input (approximate sequence embedding) from latent activation.
    activation: (latent_dim,) tensor
    projection_weights: (latent_dim x d_model) numpy or torch tensor
    Returns: (d_model,) vector
    """
    if not torch.is_tensor(projection_weights):
        projection_weights = torch.tensor(projection_weights)
    
    # Simple linear projection
    # This assumes the activation is post-bias and post-ReLU
    # This is likely an incomplete/approximate decode
    recon = torch.matmul(activation, projection_weights)
    return recon


# --- NEW STATELESS FUNCTION (from agentic_adapter.py) ---
def perturb_and_decode(
    sequence: str,
    dim: int,
    delta: float,
    esm_model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    sae_model: torch.nn.Module,
    device: str = "cpu"
) -> str:
    """
    Stateless version of the logic from agentic_adapter.RealSAE.
    
    Performs a full, per-token encode -> perturb -> decode loop.
    
    Args:
        sequence: The starting protein sequence (string).
        dim: The index of the latent feature to perturb.
        delta: The amount to add to the latent feature.
        esm_model: The trained ESM *ForMaskedLM* model (must have .lm_head).
        tokenizer: The ESM tokenizer.
        sae_model: The trained SAE model (with .encoder and .decoder).
        device: The device to run on ('cpu' or 'cuda').

    Returns:
        A new, decoded protein sequence (string).
    """
    
    # 1. ENCODE SEQUENCE (per-token)
    # token_reps shape is (L, D), where L = sequence_len + 2
    token_reps, _ = encode_sequence(sequence, esm_model, tokenizer, device=device)
    L = token_reps.shape[0]

    reconstructed_tokens = []
    
    # 2. PER-TOKEN SAE ENCODE/DECODE LOOP
    # We iterate from 1 to L-1 to skip <cls> and <eos> tokens
    for i in range(1, L - 1):
        token_vec = token_reps[i].unsqueeze(0)  # (1, hidden_dim)
        
        # 3. SAE ENCODE
        latent = sae_model.encode(token_vec)

        # 4. PERTURBATION
        # apply perturbation if requested
        if delta != 0.0 and dim < latent.shape[-1]:
            latent[0, dim] += delta

        # 5. SAE DECODE
        recon = sae_model.decode(latent).squeeze(0) # Squeeze batch dim
        reconstructed_tokens.append(recon)

    # Stack all reconstructed token embeddings
    # Shape becomes (L-2, hidden_dim)
    reconstructed = torch.stack(reconstructed_tokens).detach()

    # 6. ESM DECODE (to text)
    # Use the ESM model's LM-head to get logits from embeddings
    with torch.no_grad():
        logits = esm_model.lm_head(reconstructed)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Convert token IDs back to a string
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        
        # Filter out special tokens
        decoded_seq = "".join(
            [tok for tok in predicted_tokens if tok not in {"<cls>", "<eos>", "<pad>", "<mask>"}]
        )

    return decoded_seq

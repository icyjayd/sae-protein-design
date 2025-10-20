# utils/esm_utils.py

from transformers import EsmModel, EsmTokenizer
import torch

def load_esm2_model(model_name="facebook/esm2_t6_8M_UR50D", device="cpu"):
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def encode_sequence(sequence, model, tokenizer, device="cpu"):
    # Returns embedding (1 x L x D) and pooled representation (1 x D)
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # use the last hidden state (before LM head)
        token_reps = outputs.hidden_states[-1].squeeze(0)
    pooled = token_reps.mean(dim=0)
    return token_reps, pooled

def decode_activation(activation, projection_weights):
    """
    Reconstruct the input (approximate sequence embedding) from latent activation.
    activation: (latent_dim,) tensor
    projection_weights: (latent_dim x d_model) numpy or torch tensor
    Returns: (d_model,) vector
    """
    if not torch.is_tensor(activation):
        activation = torch.tensor(activation, dtype=torch.float32)
    if not torch.is_tensor(projection_weights):
        projection_weights = torch.tensor(projection_weights, dtype=torch.float32)
    return torch.matmul(activation, projection_weights)

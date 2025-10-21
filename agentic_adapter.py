# agentic_adapter.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from pathlib import Path
import sys, os
repo_root = os.path.abspath(os.path.dirname(__file__))
interplm_path = os.path.join(repo_root, "interplm")
if interplm_path not in sys.path:
    sys.path.insert(0, interplm_path)
from sae.utils.esm_utils import encode_sequence

from interplm.sae.inference import load_sae_from_hf

# --- SAE imports ---
from sae.extract_codes import load_mono_model_flex
# --- Scoring imports ---
from scoring import surrogate_score, predict_stability, predict_folding

OUT = Path("outputs")


class RealSAE:
    """
    RealSAE replicates the working test_reconstruction logic.
    It performs per-token encode/decode through the InterPLM SAE and reconstructs sequences.
    """

    def __init__(self, model_name: str = "esm2-8m", layer: int = 6):
        print(f"[INFO] Loading InterPLM SAE from {model_name}, layer {layer}")
        self.sae = load_sae_from_hf(model_name, plm_layer=layer)
        self.sae.eval()

        # Load the matching ESM model and tokenizer
        hf_model_name = "facebook/esm2_t6_8M_UR50D" if "8m" in model_name else "facebook/esm2_t33_650M_UR50D"
        self.model = EsmForMaskedLM.from_pretrained(hf_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.sae.to(self.device)
        self.latent_dim = getattr(self.sae, "latent_dim", 64)

    # -------------------------------------------------------------
    def encode(self, sequence: str, latent_dim: int | None = None):
        """
        Compute and return the mean latent across all tokens.
        """
        token_reps, _ = encode_sequence(sequence, self.model, self.tokenizer, device=self.device)
        L = token_reps.shape[1]
        latents = []
        for i in range(1, L - 1):
            token_vec = token_reps[0, i].unsqueeze(0)
            latent = self.sae.encode(token_vec)
            latents.append(latent.squeeze(0))
        latents = torch.stack(latents)
        z = latents.mean(dim=0).detach().cpu().numpy()
        if latent_dim:
            z = z[:latent_dim]
        return z

    # -------------------------------------------------------------
    def decode(self, sequence_or_latent):
        """
        Reconstruct a sequence from scratch (per-token SAE path).
        If input is a latent vector, it’s ignored for now —
        reconstruction follows the test_reconstruction logic.
        """
        if isinstance(sequence_or_latent, (list, np.ndarray)):
            sequence = "MKTLLILAVITAIAAGALA"  # placeholder
        else:
            sequence = sequence_or_latent

        token_reps, _ = encode_sequence(sequence, self.model, self.tokenizer, device=self.device)
        L = token_reps.shape[1]

        reconstructed_tokens = []
        for i in range(1, L - 1):
            token_vec = token_reps[0, i].unsqueeze(0)
            latent = self.sae.encode(token_vec)
            recon = self.sae.decode(latent).squeeze()
            reconstructed_tokens.append(recon)

        reconstructed = torch.stack(reconstructed_tokens).detach()
        with torch.no_grad():
            logits = self.model.lm_head(reconstructed)
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
            decoded_seq = "".join([tok for tok in predicted_tokens if tok not in {"<cls>", "<eos>", "<pad>", "<mask>"}])
        return decoded_seq

    # -------------------------------------------------------------
    def perturb_and_decode(self, sequence: str, dim: int = 0, delta: float = 0.0):
        """
        Full encode→decode reconstruction (like check_seq in test_reconstruction.py),
        with an optional perturbation of a specific latent dimension (dim, delta).
        """
        token_reps, _ = encode_sequence(sequence, self.model, self.tokenizer, device=self.device)
        token_reps = token_reps.unsqueeze(0)  # (1, L, hidden_dim)
        L = token_reps.shape[1]

        reconstructed_tokens = []
        for i in range(1, L - 1):
            token_vec = token_reps[0, i].unsqueeze(0)  # (1, hidden_dim)
            latent = self.sae.encode(token_vec)

            # apply perturbation if requested
            if delta != 0.0 and dim < latent.shape[-1]:
                latent[0, dim] += delta

            recon = self.sae.decode(latent).squeeze()
            reconstructed_tokens.append(recon)

        reconstructed = torch.stack(reconstructed_tokens).detach()  # (L-2, hidden_dim)

        with torch.no_grad():
            logits = self.model.lm_head(reconstructed)
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
            decoded_seq = "".join(
                [tok for tok in predicted_tokens if tok not in {"<cls>", "<eos>", "<pad>", "<mask>"}]
            )

        return decoded_seq

def score_sequence(sequence: str) -> Tuple[float, float, float]:
    print(f"[INFO] Scoring sequence of length {len(sequence)}")
    """Compute stability, folding, and plausibility scores."""
    stability = predict_stability(sequence)
    folding = predict_folding(sequence)
    plausibility = surrogate_score(sequence)
    return (stability, folding, plausibility)

import pytest
import numpy as np
from agentic_adapter import RealSAE

@pytest.mark.slow
def test_real_sae_decode_roundtrip():
    """
    Decode a random latent vector using the real SAE model and
    check that the output is a plausible protein sequence.
    """
    sae = RealSAE()

    # Create a random latent vector within roughly the same scale
    # as the training activations (mean 0, std ~1)
    latent = np.random.randn(sae.latent_dim).tolist()

    seq = sae.decode(latent)

    # --- Sanity checks ---
    assert isinstance(seq, str), "Decoded output must be a string"
    assert len(seq) > 0, "Decoded sequence must not be empty"
    assert set(seq).issubset(set("ACDEFGHIKLMNPQRSTVWY")), (
        f"Decoded sequence contains invalid characters: {set(seq) - set('ACDEFGHIKLMNPQRSTVWY')}"
    )

    print(f"\nDecoded sequence (first 60 aa): {seq[:60]}")
    print(f"Sequence length: {len(seq)}")

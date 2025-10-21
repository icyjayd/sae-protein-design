import pytest
import numpy as np
from agentic_adapter import RealSAE

@pytest.mark.slow
def test_real_sae_encode_shape():
    """Encode a short sequence and confirm latent vector shape is valid."""
    sae = RealSAE()
    latent = sae.encode("MKTLLILAVITAIAAGALA", latent_dim=64)
    assert isinstance(latent, list)
    assert len(latent) > 0
    assert all(isinstance(x, (float, int)) for x in latent)
    print(f"Latent vector length: {len(latent)}")
    print("First few values:", np.array(latent)[:5])

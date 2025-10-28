from sae.utils import esm_utils

def decode_with_deltas(sequence, dZ, sae_model, esm_model, tokenizer, device, threshold):
    L, N = dZ.shape
    perturb = {}
    dZ_cpu = dZ.detach().cpu().numpy()
    for i in range(L):
        edits = {j: float(dZ_cpu[i, j]) for j in range(N) if abs(dZ_cpu[i, j]) > threshold}
        if edits:
            perturb[i+1] = edits
    seq_pert = esm_utils.perturb_and_decode(
        sequence=sequence,
        sae_model=sae_model,
        esm_model=esm_model,
        tokenizer=tokenizer,
        surgical_perturbations=perturb,
        device=device,
    )
    return seq_pert

def get_activation_matrix(sequence, sae_model, esm_model, tokenizer, device):
    return esm_utils.get_activation_matrix(sequence, sae_model, esm_model, tokenizer, device=device)

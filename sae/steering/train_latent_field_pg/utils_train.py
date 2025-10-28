import torch, numpy as np
from sae.utils import esm_utils
def quantize_m(m_vals, m_min, m_max, bins):
    edges = np.linspace(m_min, m_max, int(bins))
    idxs = np.abs(edges[None, :] - np.array(m_vals)[:, None]).argmin(-1)
    return edges[idxs].astype(float)
def get_Z_cached(seq, sae_model, esm_model, tokenizer, device, act_cache):
    if act_cache is not None:
        z = act_cache.get(seq)
        if z is not None:
            return z
    z = esm_utils.get_activation_matrix(seq, sae_model, esm_model, tokenizer, device=device)
    if act_cache is not None:
        act_cache.set(seq, z.cpu())
    return z
def decode_batched_with_cache(seq_batch, dZ_batch, sae_model, esm_model, tokenizer,
                              device, threshold, m_bins_for_key, dec_cache, use_mean):
    mode = 'mean' if use_mean else 'sample'
    decoded = [None] * len(seq_batch)
    need_idx, need_seqs, need_dZ = [], [], []
    for i, (s, mbin) in enumerate(zip(seq_batch, m_bins_for_key)):
        cached = dec_cache.get(s, mbin, mode) if dec_cache is not None else None
        if cached is None:
            need_idx.append(i); need_seqs.append(s); need_dZ.append(dZ_batch[i])
        else:
            decoded[i] = cached
    if need_idx:
        pert = esm_utils.build_perturbations_from_batch(need_dZ, threshold=threshold)
        dec_out = esm_utils.perturb_and_decode_batch(need_seqs, sae_model, esm_model, tokenizer,
                                                     pert, device=device, threshold=threshold, dtype=torch.float16)
        for k, seq_pert in zip(need_idx, dec_out):
            decoded[k] = seq_pert
            if dec_cache is not None:
                dec_cache.set(seq_batch[k], float(m_bins_for_key[k]), mode, seq_pert)
    return decoded

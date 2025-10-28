import torch, numpy as np
from sae.steering.train_latent_field_pg.utils_train import get_Z_cached, quantize_m, decode_batched_with_cache
from sae.steering.pg.surrogate_utils import predict_sequences

def train_epoch_batched(policy, batch, sae_model, esm_model, tokenizer, ridge, device,
                        encoding, m_range, m_bins, threshold, entropy_coef, sparse_coef,
                        baseline, baseline_beta, act_cache, enc_cache, dec_cache,
                        use_mean_for_reward):
    seq_batch, label_batch = batch
    m_vals = np.random.uniform(m_range[0], m_range[1], len(seq_batch)).astype(np.float32)
    m_torch = torch.tensor(m_vals, dtype=torch.float32, device=device)

    # You can quantize or discretize m here if you want to actually use bins:
    if m_bins > 1:
        bins = np.linspace(m_range[0], m_range[1], m_bins)
        m_torch = torch.tensor(
            [min(bins, key=lambda b: abs(float(m_i) - b)) for m_i in m_torch],
            dtype=torch.float32, device=device
        )

    Z_list = [get_Z_cached(s, sae_model, esm_model, tokenizer, device, act_cache)
              for s in seq_batch]

    losses = []
    for Z_i, s, m_i in zip(Z_list, seq_batch, m_torch):
        dZ, y_hat = policy(Z_i, m_i)
        seq_new = decode_batched_with_cache([s], [dZ], sae_model, esm_model, tokenizer,
                                            device, threshold, [m_i], dec_cache, use_mean_for_reward)[0]
        y0 = predict_sequences(ridge, [s], encoding, encoding_cache=enc_cache)[0]
        y1 = predict_sequences(ridge, [seq_new], encoding, encoding_cache=enc_cache)[0]
        y_delta_true = y1 - y0
        L_score = (y_hat.squeeze() - torch.tensor(y_delta_true, device=device))**2
        reward = -L_score.detach()
        L_policy = -(reward - baseline) * torch.mean(y_hat)
        loss = L_score + L_policy + sparse_coef * dZ.abs().mean()
        losses.append(loss)

    total_loss = torch.stack(losses).mean()
    baseline = (1 - baseline_beta) * baseline + baseline_beta * float(reward.mean())
    return total_loss, baseline

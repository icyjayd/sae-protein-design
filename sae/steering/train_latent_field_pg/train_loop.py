import torch, csv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from sae.steering.train_latent_field_pg.epoch_step import train_epoch_batched
from sae.steering.train_latent_field_pg.cache_helpers import cache_coverage
from sae.steering.pg.surrogate_utils import predict_sequences

def train_loop(policy, dataloader, sae_model, esm_model, tokenizer, ridge,
               device, encoding, m_range, threshold, entropy_coef, sparse_coef,
               baseline_beta, epochs, print_every, eval_every,
               act_cache, enc_cache, dec_cache,
               use_mean_for_reward, m_bins,
               train_x, train_y, test_x, test_y,
               persist_caches=False, outdir="."):

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    baseline = 0.0
    log_path = Path(outdir) / "epoch_metrics.csv"

    # Initialize CSV header
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "avg_loss", "baseline",
                "spearman_train", "p_train", "r2_train", "mse_train",
                "spearman_test", "p_test", "r2_test", "mse_test"
            ])

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            loss, baseline = train_epoch_batched(
                policy, batch, sae_model, esm_model, tokenizer, ridge, device,
                encoding, m_range, m_bins, threshold, entropy_coef, sparse_coef,
                baseline, baseline_beta, act_cache, enc_cache, dec_cache, use_mean_for_reward
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch} avg_loss {avg_loss:.4f} baseline {baseline:.4f}")

        # --- Evaluation ---
        if epoch % eval_every == 0:
            y_pred_train = predict_sequences(ridge, train_x, encoding, encoding_cache=enc_cache)
            y_pred_test = predict_sequences(ridge, test_x, encoding, encoding_cache=enc_cache)

            rho_tr, p_tr = spearmanr(train_y, y_pred_train)
            rho_te, p_te = spearmanr(test_y, y_pred_test)
            r2_tr = r2_score(train_y, y_pred_train)
            r2_te = r2_score(test_y, y_pred_test)
            mse_tr = mean_squared_error(train_y, y_pred_train)
            mse_te = mean_squared_error(test_y, y_pred_test)

            print(
                f"[eval] epoch {epoch}: "
                f"Spearman_train={rho_tr:.3f}, R2_train={r2_tr:.3f}, MSE_train={mse_tr:.3f} | "
                f"Spearman_test={rho_te:.3f}, R2_test={r2_te:.3f}, MSE_test={mse_te:.3f}"
            )

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, avg_loss, baseline,
                    rho_tr, p_tr, r2_tr, mse_tr,
                    rho_te, p_te, r2_te, mse_te
                ])

        # --- Cache diagnostics & save ---
        if persist_caches:
            for name, c, seqs in [
                ("activations", act_cache, train_x),
                ("encodings", enc_cache, train_x),
                ("decodes", dec_cache, train_x)
            ]:
                if c:
                    n, frac = cache_coverage(c, seqs)
                    print(f"[cache] {name}: {n}/{len(seqs)} cached ({frac*100:.1f}%)")
                    c.save()
            print(f"[cache] Saved caches after epoch {epoch}")

    return policy

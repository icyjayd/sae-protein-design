import argparse, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from sae.utils import esm_utils
from sae.steering.pg.data_utils import GeneralDataset, load_data, split_dataset
from sae.steering.pg.surrogate_utils import load_or_train_surrogate, predict_sequences
from sae.steering.pg.policy import StochasticLatentPolicy
from sae.steering.pg.perturb_utils import decode_with_deltas, get_activation_matrix
from sae.steering.pg.eval_utils import log_eval

def train_policy(policy, dataloader, sae_model, esm_model, tokenizer, ridge,
                 device, encoding, m_min, m_max, threshold,
                 entropy_coef, sparse_coef, baseline_beta, epochs, print_every,
                 train_x, train_y, test_x, test_y, eval_every):
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    baseline = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        for seq, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
            seq = seq[0]
            m_val = np.random.uniform(m_min, m_max)
            m = torch.tensor(m_val, dtype=torch.float32, device=device)

            Z = get_activation_matrix(seq, sae_model, esm_model, tokenizer, device)
            dZ, dZ_mean, logp, ent = policy.sample(Z, m)
            seq_pert = decode_with_deltas(seq, dZ, sae_model, esm_model, tokenizer, device, threshold)

            y0 = predict_sequences(ridge, [seq], encoding)[0]
            yp = predict_sequences(ridge, [seq_pert], encoding)[0]

            reward = -((yp - (y0 + m_val)) ** 2)
            baseline = (1 - baseline_beta) * baseline + baseline_beta * reward
            adv = reward - baseline

            sparse_pen = sparse_coef * dZ.abs().mean()
            loss = -adv * logp + sparse_pen - (entropy_coef * ent)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        if epoch % print_every == 0:
            print(f"epoch {epoch} avg_obj {total_loss/len(dataloader):.4f} baseline {baseline:.4f}")
        if epoch % eval_every == 0:
            log_eval(epoch, ridge, encoding, train_x, train_y, test_x, test_y)
    return policy

def sweep_codirectionality(seq, policy, sae_model, esm_model, tokenizer,
                           ridge, device, encoding, m_start, m_end, m_steps, threshold):
    mags = np.linspace(m_start, m_end, int(m_steps))
    results = []
    y0 = predict_sequences(ridge, [seq], encoding)[0]
    Z = get_activation_matrix(seq, sae_model, esm_model, tokenizer, device)
    for m_val in mags:
        m = torch.tensor(m_val, dtype=torch.float32, device=device)
        with torch.no_grad():
            dZ_mean = policy(Z, m)
        seq_pert = decode_with_deltas(seq, dZ_mean, sae_model, esm_model, tokenizer, device, threshold)
        yp = predict_sequences(ridge, [seq_pert], encoding)[0]
        results.append((float(m_val), float(yp - y0)))
    return results

def main():
    parser = argparse.ArgumentParser(description="Train latent field with REINFORCE using ridge surrogate (modular).")
    parser.add_argument("--seqs", required=True)
    parser.add_argument("--scores", required=False)
    parser.add_argument("--sae", required=False)
    parser.add_argument("--ridge", required=True)
    parser.add_argument("--encoding", type=str, default="aac")
    parser.add_argument("--esm", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="runs/poc_latent_field_pg/")
    parser.add_argument("--m-range", nargs=3, type=float, default=[-1.0, 1.0, 11])
    parser.add_argument("--train-epochs", type=int, default=150)
    parser.add_argument("--policy-std", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=1e-4)
    parser.add_argument("--sparse-coef", type=float, default=1e-3)
    parser.add_argument("--baseline-beta", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--interplm_layer", type=int, default=6)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate (and print Spearman train/test) every N epochs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seqs, labels, splits = load_data(args.seqs, args.scores)
    (train_x, train_y), (test_x, test_y) = split_dataset(seqs, labels, splits)
    dataloader = DataLoader(GeneralDataset(train_x, train_y), batch_size=1, shuffle=True)

    esm_model, tokenizer = esm_utils.load_esm2_model(args.esm, device=args.device)
    if args.sae:
        print(f"[INFO] Loading custom SAE from {args.sae}")
        sae_model = torch.load(args.sae, map_location=args.device)
    else:
        print(f"[INFO] Loading InterPLM SAE corresponding to {args.esm}, layer {args.interplm_layer}")
        sae_model = esm_utils.load_interplm(args.esm, plm_layer=args.interplm_layer, device=args.device)

    ridge = load_or_train_surrogate(args.ridge, train_x, train_y, args.encoding)

    latent_dim = sae_model.decoder.out_features
    policy = StochasticLatentPolicy(latent_dim=latent_dim, sigma=args.policy_std).to(args.device)

    policy = train_policy(policy, dataloader, sae_model, esm_model, tokenizer, ridge,
                          args.device, args.encoding, args.m_range[0], args.m_range[1],
                          args.threshold, args.entropy_coef, args.sparse_coef,
                          args.baseline_beta, args.train_epochs, args.print_every,
                          train_x, train_y, test_x, test_y, args.eval_every)

    results = sweep_codirectionality(test_x[0], policy, sae_model, esm_model, tokenizer,
                                     ridge, args.device, args.encoding,
                                     args.m_range[0], args.m_range[1], args.m_range[2],
                                     args.threshold)

    with open(os.path.join(args.outdir, "codirectionality_pg.json"), "w") as f:
        json.dump(results, f, indent=2)

    mags, deltas = zip(*results)
    plt.figure(figsize=(5,4))
    plt.plot(mags, deltas, marker="o")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Requested magnitude (m)")
    plt.ylabel("Predicted property change (dy)")
    plt.title("Codirectionality Sweep (Policy Mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "codirectionality_pg.png"))
    plt.close()

    torch.save(policy.state_dict(), os.path.join(args.outdir, "latent_policy_pg.pth"))
    print(f"✅ Training complete. Results saved in {args.outdir}")

if __name__ == "__main__":
    main()

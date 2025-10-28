import os, torch
from torch.utils.data import DataLoader
from sae.steering.pg.data_utils import GeneralDataset
from sae.steering.train_latent_field_pg.train_loop import train_loop

def run_training(args, train_x, train_y, test_x, test_y,
                 sae_model, esm_model, tokenizer,
                 ridge, policy, act_cache, enc_cache, dec_cache):
    dl = DataLoader(GeneralDataset(train_x, train_y),
                    batch_size=args.batch_size, shuffle=True)
    policy = train_loop(policy, dl, sae_model, esm_model, tokenizer, ridge,
                        args.device, args.encoding, args.m_range,
                        args.threshold, args.entropy_coef, args.sparse_coef,
                        args.baseline_beta, args.train_epochs,
                        args.print_every, args.eval_every,
                        act_cache, enc_cache, dec_cache,
                        args.use_mean_for_reward, args.m_bins,
                        train_x, train_y, test_x, test_y,
                        persist_caches=args.persist_caches, outdir=args.outdir)
    torch.save(policy.state_dict(), os.path.join(args.outdir, "latent_policy_hybrid.pth"))
    print(f"✅ Model saved to {args.outdir}")

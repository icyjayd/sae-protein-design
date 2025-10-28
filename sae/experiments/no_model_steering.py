import argparse, os, pandas as pd, numpy as np
from sae.experiments.no_model_steering_modules.steering_utils import reconstruct_and_steer
from sae.experiments.no_model_steering_modules.ridge_utils import score_with_ridge, summarize
from sae.experiments.no_model_steering_modules.cache_utils import load_cache, update_cache
from sae.experiments.no_model_steering_modules.parallel_utils import parallel_latent_runs
from sae.utils.esm_utils import load_esm2_model, load_interplm

def sample_train_test(df:pd.DataFrame,sample_size:int=100)->pd.DataFrame:
    train_df=df[df["split"]=="train"]
    test_df=df[df["split"]=="test"]
    train_df=train_df.sample(sample_size,random_state=42)
    test_df=test_df.sample(sample_size,random_state=42)
    df=pd.concat([train_df,test_df]).reset_index(drop=True)
    return df
def run_single_latent(args, latent_idx:int):
    esm_model, tokenizer = load_esm2_model(args.esm, device=args.device)
    sae_model = load_interplm(args.esm, plm_layer=args.plm_layer, device=args.device)
    df = pd.read_csv(args.sequences)
    cache = load_cache(args.cache_file)
    new_rows=[]
    for _,row in df.iterrows():

        seq=row["sequence_in"] if "sequence_in" in row else row["sequence"]
        split=row["split"]
        if not cache.empty and ((cache["sequence_in"]==seq)&(cache["latent_idx"]==latent_idx)
                                &(cache["magnitude"]==args.delta)&(cache["mode"]==args.mode)).any():
            continue
        df_steer = reconstruct_and_steer(seq, sae_model, esm_model, tokenizer,
                                         latent_idx, args.delta, args.mode,
                                         args.chosen_pos, args.filter_kind,
                                         args.filter_value, args.n_random,
                                         args.seed, args.device)
        df_steer["split"]=split
        df_steer["latent_idx"]=latent_idx
        df_steer["magnitude"]=args.delta
        df_steer["mode"]=args.mode
        new_rows.append(df_steer)
    if not new_rows:
        print(f"[latent {latent_idx}] No new sequences to process.")
        return
    df_new=pd.concat(new_rows,ignore_index=True)
    y_r,y_s=score_with_ridge(args.ridge_model,df_new["sequence_recon"],df_new["sequence_steer"],encoding=args.encoding)
    
    df_new["y_recon"]=y_r; df_new["y_steer"]=y_s
    df_new["delta"]=y_s - y_r; df_new["abs_delta"]=abs(df_new["delta"])
    df_new["no_change"]=(df_new["sequence_recon"]==df_new["sequence_steer"]).astype(int)
    df_sum=summarize(df_new)
    tag=f"lat{latent_idx}_d{args.delta}_{args.mode}"
    rows_csv=os.path.join(args.outdir,f"rows_{args.property}_{tag}.csv")
    sum_csv=os.path.join(args.outdir,f"summary_{args.property}_{tag}.csv")
    df_new.to_csv(rows_csv,index=False); df_sum.to_csv(sum_csv,index=False)
    update_cache(args.cache_file,df_new)

def main():
    p=argparse.ArgumentParser(description="Generate & score ridge experiments with caching & parallel latents")
    p.add_argument("--sequences",required=True); p.add_argument("--ridge_model",required=True)
    p.add_argument("--property",default="property"); p.add_argument("--mode",required=True)
    p.add_argument("--delta",type=float,required=True); p.add_argument("--latent_idx",type=int)
    p.add_argument("--latent_ranking_csv"); p.add_argument("--ranking_column",default="spearman")
    p.add_argument("--top_n_latents",type=int,default=5); p.add_argument("--n_workers",type=int,default=4)
    p.add_argument("--encoding",default="onehot"); p.add_argument("--filter_kind"); p.add_argument("--filter_value",type=float)
    p.add_argument("--chosen_pos",type=int); p.add_argument("--n_random",type=int,default=1)
    p.add_argument("--seed",type=int,default=0); p.add_argument("--outdir",default="analysis_results")
    p.add_argument("--cache_file",default="analysis_results/cache_all_runs.csv")
    p.add_argument("--esm",default="facebook/esm2_t6_8M_UR50D"); p.add_argument("--plm_layer",type=int,default=6)
    import torch; p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args(); os.makedirs(args.outdir,exist_ok=True)
    if args.latent_idx is not None:
        latent_indices=[args.latent_idx]; ranking_df=None
    else:
        df_lat=pd.read_csv(args.latent_ranking_csv); df_lat["abs_corr"]=df_lat[args.ranking_column].abs()
        latent_indices=(df_lat.sort_values("abs_corr",ascending=False)
                            .head(args.top_n_latents)["latent_index"].astype(int).tolist())
        ranking_df=df_lat
        print(f"[INFO] Running top {len(latent_indices)} latents: {latent_indices}")
    if len(latent_indices)>1:
        parallel_latent_runs(latent_indices, run_single_latent, args, args.n_workers)
        import glob
        from scipy.stats import spearmanr
        files = glob.glob(os.path.join(args.outdir, f"summary_{args.property}_lat*_*.csv"))
        if files:
            all_sum = pd.concat(
                [pd.read_csv(f).assign(latent_index=int(f.split("lat")[1].split("_")[0])) for f in files],
                ignore_index=True
            )

            # Merge abs_corr from ranking_df if available
            if ranking_df is not None and "latent_index" in ranking_df.columns:
                all_sum = all_sum.merge(
                    ranking_df[["latent_index", "abs_corr"]],
                    on="latent_index",
                    how="left"
                )

            # --- Cross-latent Spearman analyses ---
            if "spearman_yrecon_vs_ysteer" in all_sum.columns and "abs_corr" in all_sum.columns:
                try:
                    rho, pval = spearmanr(all_sum["abs_corr"], all_sum["spearman_yrecon_vs_ysteer"], nan_policy="propagate")
                    all_sum["spearman_latent_vs_yrecon_ysteer"] = rho
                    all_sum["pval_latent_vs_yrecon_ysteer"] = pval
                    print(f"[INFO] abs_corr ↔ spearman_yrecon_vs_ysteer: ρ={rho:.3f}, p={pval:.3g}")
                except Exception as e:
                    print(f"[WARN] Failed latent-wise correlation (y_recon vs y_steer): {e}")
                    all_sum["spearman_latent_vs_yrecon_ysteer"] = np.nan
                    all_sum["pval_latent_vs_yrecon_ysteer"] = np.nan

            if "spearman_abs_yrecon_vs_abs_delta" in all_sum.columns and "abs_corr" in all_sum.columns:
                try:
                    rho2, pval2 = spearmanr(all_sum["abs_corr"], all_sum["spearman_abs_yrecon_vs_abs_delta"], nan_policy="propagate")
                    all_sum["spearman_latent_vs_abs_yrecon_abs_delta"] = rho2
                    all_sum["pval_latent_vs_abs_yrecon_abs_delta"] = pval2
                    print(f"[INFO] abs_corr ↔ spearman_abs_yrecon_vs_abs_delta: ρ={rho2:.3f}, p={pval2:.3g}")
                except Exception as e:
                    print(f"[WARN] Failed latent-wise abs correlation: {e}")
                    all_sum["spearman_latent_vs_abs_yrecon_abs_delta"] = np.nan
                    all_sum["pval_latent_vs_abs_yrecon_abs_delta"] = np.nan

            out_path = os.path.join(args.outdir, f"summary_all_latents_{args.property}.csv")
            all_sum.to_csv(out_path, index=False)
            print(f"[INFO] Combined summary saved to {out_path}")
        else:
            print("[WARN] No summary files found for combination step.")
    else:
        run_single_latent(args,latent_indices[0])

if __name__=="__main__": main()

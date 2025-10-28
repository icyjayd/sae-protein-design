from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def parallel_latent_runs(latent_indices, func, args, n_workers:int):
    print(f"[INFO] Running {len(latent_indices)} latents on {n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures=[ex.submit(func,args,li) for li in latent_indices]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="Latent experiments", unit="latent"):
            pass
    print("[INFO] All latent runs complete.")

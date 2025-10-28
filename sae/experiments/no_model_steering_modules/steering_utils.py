import torch, random
import pandas as pd
from typing import List
from sae.utils.esm_utils import load_esm2_model, load_interplm, get_activation_matrix, perturb_and_decode

def get_filtered_positions(acts: torch.Tensor, latent_idx: int, kind: str, val: float) -> List[int]:
    col = acts[:, latent_idx].detach().cpu()
    L = col.shape[0]
    if kind == "threshold":
        keep = [i+1 for i,v in enumerate(col) if float(v)>=val]
    elif kind == "percentile":
        q = min(max(val/100.0,0),1)
        thr = float(torch.quantile(col,q))
        keep = [i+1 for i,v in enumerate(col) if float(v)>=thr]
    elif kind == "topk":
        k = max(1, min(int(val), L))
        idx = torch.topk(col,k=k).indices.cpu().tolist()
        keep = [i+1 for i in idx]
    elif kind == "sign":
        sgn = 1 if val>=0 else -1
        keep = [i+1 for i,v in enumerate(col) if (float(v)>=0)==(sgn>0)]
    else:
        raise ValueError(kind)
    return sorted(keep)

def reconstruct_and_steer(seq, sae_model, esm_model, tokenizer,
                          latent_idx:int, delta:float, mode:str,
                          chosen_pos:int=None, filter_kind:str=None,
                          filter_value:float=None, n_random:int=1, seed:int=0,
                          device:str="cpu") -> pd.DataFrame:
    rng = random.Random(seed)
    seq_recon = perturb_and_decode(seq, sae_model, esm_model, tokenizer,
                                   surgical_perturbations=None, device=device)
    Z = get_activation_matrix(seq, sae_model, esm_model, tokenizer, device=device)
    L = Z.shape[0]
    rows=[]
    if mode=="global_latent":
        edits={p:{latent_idx:delta} for p in range(1,L+1)}
        residues="all"
        seq_steer=perturb_and_decode(seq,sae_model,esm_model,tokenizer,surgical_perturbations=edits,device=device)
        rows.append((seq,seq_recon,seq_steer,residues))
    elif mode=="random_residue":
        for _ in range(n_random):
            pos=rng.randint(1,L)
            edits={pos:{latent_idx:delta}}
            residues=str(pos)
            seq_steer=perturb_and_decode(seq,sae_model,esm_model,tokenizer,surgical_perturbations=edits,device=device)
            rows.append((seq,seq_recon,seq_steer,residues))
    elif mode=="chosen_residue":
        if chosen_pos is None: raise ValueError("--chosen_pos required")
        edits={chosen_pos:{latent_idx:delta}}; residues=str(chosen_pos)
        seq_steer=perturb_and_decode(seq,sae_model,esm_model,tokenizer,surgical_perturbations=edits,device=device)
        rows.append((seq,seq_recon,seq_steer,residues))
    elif mode=="filtered_residues":
        if not filter_kind or filter_value is None: raise ValueError("filter params required")
        keep=get_filtered_positions(Z,latent_idx,filter_kind,filter_value)
        edits={p:{latent_idx:delta} for p in keep}
        residues=",".join(map(str,keep))
        seq_steer=perturb_and_decode(seq,sae_model,esm_model,tokenizer,surgical_perturbations=edits,device=device)
        rows.append((seq,seq_recon,seq_steer,residues))
    return pd.DataFrame(rows,columns=["sequence_in","sequence_recon","sequence_steer","residues"])

import argparse,json
from .train import train_model
def main():
    ap=argparse.ArgumentParser(description="Train ML models for protein sequence scoring")
    ap.add_argument("sequences"); ap.add_argument("--labels"); ap.add_argument("--model",default="xgb")
    ap.add_argument("--encoding",default="kmer",choices=["aac","kmer","onehot","esm"])
    ap.add_argument("--task",choices=["classification","regression"]); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--outdir",default="runs/ml_model"); ap.add_argument("--k",type=int,default=3); ap.add_argument("--max-len",type=int,default=512)
    a=ap.parse_args()
    res=train_model(a.sequences,a.labels,a.model,a.encoding,a.task,a.seed,a.outdir,k=a.k,max_len=a.max_len)
    print(json.dumps(res,indent=2))
if __name__=="__main__": main()

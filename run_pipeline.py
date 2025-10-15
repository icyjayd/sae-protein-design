#!/usr/bin/env python3
import argparse, sys, subprocess, json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from utils.model_utils import MonosemanticSAE, SparseAutoencoderSAE, get_device
from datetime import datetime
sys.path.append("interplm")
from interplm.sae.inference import load_sae_from_hf

OUT = Path("outputs")

INTERPLM_PRESETS = {
    "esm2-8m": {"default": 4, "layers": [2, 4, 6]},
    "esm2-35m": {"default": 12, "layers": [6, 12, 20]},
    "esm2-150m": {"default": 24, "layers": [12, 24, 36]},
    "esm2-650m": {"default": 32, "layers": [16, 32, 48]},
}

def list_presets():
    print("Available Hugging Face InterPLM presets:\n")
    for model, cfg in INTERPLM_PRESETS.items():
        print(f"- {model}: default={cfg['default']} layers={cfg['layers']}")
    sys.exit(0)

def run_cmd(cmd, env=None):
    print("[RUN]", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(res.stdout)
    if res.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")

def ensure_interplm_installed():
    try:
        import interplm
        return True
    except Exception:
        print("[INFO] interplm not found. Installing...")
        res = subprocess.run([sys.executable, "-m", "pip", "install", "interplm"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res.stdout)
        try:
            import interplm
            return True
        except Exception:
            print("[WARN] Failed to install interplm automatically.")
            return False

def load_hf_sae(plm_model: str, plm_layer: int):
    ok = ensure_interplm_installed()
    if not ok:
        raise SystemExit("interplm is required for --from_hf_model usage. Please install it and retry.")
    return load_sae_from_hf(plm_model=plm_model, plm_layer=plm_layer)

def freeze_modules(model: torch.nn.Module, freeze: str, lr_mult: float = 0.1):
    params = []
    def set_requires(prefix, req):
        for n, p in model.named_parameters():
            if n.startswith(prefix):
                p.requires_grad = req
    if freeze == "encoder":
        set_requires("encoder", False)
    elif freeze == "decoder":
        set_requires("decoder", False)
    elif freeze == "both":
        set_requires("encoder", False); set_requires("decoder", False)
    for n, p in model.named_parameters():
        params.append({"params": [p], "lr_mult": (lr_mult if not p.requires_grad else 1.0)})
    return params

def finetune(model, X, epochs=10, lr=1e-3, l1=1e-3, device=None, freeze="none", lr_mult=0.1):
    device = get_device(device); model.to(device)
    groups = freeze_modules(model, freeze, lr_mult=lr_mult)
    opt = torch.optim.Adam([{"params": g["params"], "lr": lr * g["lr_mult"]} for g in groups])
    mse = torch.nn.MSELoss()
    X = torch.from_numpy(X.astype("float32")).to(device)
    for ep in range(1, epochs+1):
        opt.zero_grad()
        recon, z = model(X)
        loss = mse(recon, X) + l1 * torch.mean(torch.abs(z))
        loss.backward(); opt.step()
        if ep % 2 == 0 or ep == 1:
            print(f"[finetune] epoch {ep}/{epochs} loss={loss.item():.6f}")
    return model

def build_report(pdf_path: Path):
    metadata = {}
    meta_path = OUT / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11)); plt.axis('off')
        plt.text(0.1, 0.95, "SAE Steering Pipeline Report", fontsize=20, weight='bold')
        if metadata:
            y = 0.91
            if "gene" in metadata:
                plt.text(0.1, y, f"Gene: {metadata['gene']}", fontsize=12); y -= 0.03
            if "property" in metadata:
                plt.text(0.1, y, f"Property: {metadata['property']}", fontsize=12); y -= 0.03
        pdf.savefig(fig); plt.close(fig)

        for suffix, label in (("regular","Regular"), ("mono","Monosemantic")):
            p = OUT / f"latent_property_corr_{suffix}.npy"
            if p.exists():
                corrs = np.load(p)
                fig = plt.figure(figsize=(8.5, 4))
                plt.title(f"Latent–Property Correlations ({label})")
                plt.plot(corrs, marker='o', linestyle='none', markersize=3)
                plt.xlabel("latent dim"); plt.ylabel("Spearman rho")
                pdf.savefig(fig); plt.close(fig)
            p = OUT / f"active_learning_{suffix}.npy"
            if p.exists():
                arr = np.load(p)
                fig = plt.figure(figsize=(8.5, 4)); plt.title(f"Active Learning ({label})")
                plt.plot(arr, marker='o'); plt.xlabel("round"); plt.ylabel("best score")
                pdf.savefig(fig); plt.close(fig)
            p = OUT / f"smoothness_{suffix}.npy"
            if p.exists():
                arr = np.load(p)
                fig = plt.figure(figsize=(8.5, 4)); plt.title(f"Smoothness ({label})")
                plt.plot(arr); plt.xlabel("step"); plt.ylabel("predicted"); pdf.savefig(fig); plt.close(fig)
            p = OUT / f"atoms_pca_{suffix}.npy"
            if p.exists():
                xy = np.load(p)
                fig = plt.figure(figsize=(6,6)); plt.title(f"Atoms PCA ({label})")
                plt.scatter(xy[:,0], xy[:,1], s=10); plt.xlabel("PC1"); plt.ylabel("PC2"); pdf.savefig(fig); plt.close(fig)

        sp = OUT / "metrics_summary.json"
        fig = plt.figure(figsize=(8.5, 11)); plt.axis('off')
        if sp.exists():
            data = json.loads(sp.read_text())
            y = 0.95
            plt.text(0.1, y, "Summary", fontsize=16, weight='bold'); y -= 0.04
            for m in data.get("models", []):
                line = f"- {m['suffix']}: R2={m['surrogate_r2']:.3f}, mean rho={m['corr_mean']:.3f}, sparsity={m['sparsity_fraction']:.2f}, diversity={m['diversity_global']}"
                plt.text(0.1, y, line, fontsize=11); y -= 0.03
        else:
            plt.text(0.1, 0.95, "No metrics_summary.json found.", fontsize=12)
        pdf.savefig(fig); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-hf-presets", action="store_true")
    ap.add_argument("--from_hf_model", type=str, default="", help="e.g., esm2-8m")
    ap.add_argument("--layer", type=int, default=None, help="PLM layer index for HF model")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--freeze", type=str, default="none", choices=["none","encoder","decoder","both"])
    ap.add_argument("--lr-mult", type=float, default=0.1)
    ap.add_argument("--mode", choices=["regular","monosemantic","both"], default="both")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--l1", type=float, default=1e-3)
    ap.add_argument("--decor", type=float, default=5e-3)
    ap.add_argument("--ortho", type=float, default=1e-3)
    ap.add_argument("--unit", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--threshold-pct", type=int, default=70)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--gene", type=str, default="", help="Gene name (optional)")
    ap.add_argument("--property", type=str, default="", help="Property measured (optional)")
    args = ap.parse_args()

    if args.list_hf_presets:
        list_presets()

    if args.gene or args.property:
        (OUT / "metadata.json").write_text(json.dumps({
            "gene": args.gene,
            "property": args.property,
            "model": args.from_hf_model or "retrained",
            "layer": args.layer,
            "mode": args.mode,
            "retrain": args.retrain,
            "timestamp": datetime.now().isoformat()

        }, indent=2))

    if not (OUT / "activations.npy").exists():
        raise SystemExit("Missing outputs/activations.npy. Run extract_from_hf_model.py or generate_activations.py first.")

    used_hf = False
    if args.from_hf_model:
        if args.from_hf_model not in INTERPLM_PRESETS:
            raise SystemExit(f"Unknown model {args.from_hf_model}. Run with --list-hf-presets.")
        if args.layer is None:
            args.layer = INTERPLM_PRESETS[args.from_hf_model]["default"]
            print(f"[INFO] Using default layer {args.layer} for {args.from_hf_model}")
        print(f"[INFO] Loading HF pretrained SAE for {args.from_hf_model} layer {args.layer}")
        sae = load_hf_sae(args.from_hf_model, args.layer)
        device = get_device(args.device); sae.to(device)
        acts = np.load(OUT / "activations.npy")
        input_dim = acts.shape[1]
        if args.finetune:
            print(f"[INFO] Fine-tuning HF SAE (freeze={args.freeze}, lr-mult={args.lr_mult}, epochs={args.epochs})")
            sae = finetune(sae, acts, epochs=max(1,args.epochs), lr=1e-3, l1=args.l1, device=args.device, freeze=args.freeze, lr_mult=args.lr_mult)
            tag = f"mono_finetuned_{args.from_hf_model}_layer{args.layer}"
        else:
            tag = f"mono_{args.from_hf_model}_layer{args.layer}"
        torch.save(sae.state_dict(), OUT / f"sae_{tag}.pt")
        cfg = {"input_dim": int(input_dim), "latent_dim": getattr(sae, 'latent_dim', 10240), "hidden": getattr(sae, 'hidden', 512), "topk": getattr(sae, 'topk', None)}
        (OUT / f"sae_{tag}_config.json").write_text(json.dumps(cfg, indent=2))
        torch.save(sae.state_dict(), OUT / "sae_mono.pt")
        (OUT / "sae_mono_config.json").write_text(json.dumps(cfg, indent=2))
        used_hf = True

    if not used_hf and args.retrain:
        run_cmd([sys.executable, "train.py", "--mode", args.mode,
                 "--epochs", str(args.epochs), "--latent-dim", str(args.latent_dim),
                 "--hidden", str(args.hidden), "--l1", str(args.l1),
                 "--decor", str(args.decor), "--ortho", str(args.ortho),
                 "--unit", str(args.unit), "--topk", str(args.topk),
                 "--device", str(args.device or "")])
    elif not used_hf:
        print("[INFO] Skipping training (no --retrain and no --from_hf_model). Expect existing models in outputs/.")

    run_cmd([sys.executable, "extract_codes.py", "--mode", "monosemantic" if used_hf else args.mode, "--threshold-pct", str(args.threshold_pct)])
    run_cmd([sys.executable, "analysis_metrics.py"], env=dict(**os.environ, GENE=args.gene, PROPERTY=args.property))
    build_report(OUT / "pipeline_report.pdf")

if __name__ == "__main__":
    main()

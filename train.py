
#!/usr/bin/env python3
import argparse, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from utils.model_utils import SparseAutoencoderSAE, MonosemanticSAE, get_device

OUT = Path('outputs'); OUT.mkdir(exist_ok=True)

def _train(model, X, epochs=60, batch_size=128, lr=1e-3, l1_coef=1e-3,
           decor=0.0, ortho=0.0, unit=0.0, device=None):
    device = get_device(device); model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X.astype('float32'))), batch_size=batch_size, shuffle=True)
    for ep in range(1, epochs+1):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device); opt.zero_grad()
            recon, z = model(xb)
            loss = mse(recon, xb) + l1_coef * torch.mean(torch.abs(z))
            if hasattr(model, 'latent_decorrelation_loss') and decor>0: loss += decor * model.latent_decorrelation_loss(z)
            if hasattr(model, 'decoder_orthonormal_loss') and ortho>0: loss += ortho * model.decoder_orthonormal_loss()
            if hasattr(model, 'decoder_unitnorm_loss') and unit>0: loss += unit * model.decoder_unitnorm_loss()
            loss.backward(); opt.step(); total += loss.item() * xb.size(0)
        if ep % 10 == 0 or ep == 1: print(f'Epoch {ep}/{epochs} loss={total/len(dl.dataset):.6f}')
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['regular','monosemantic','both'], default='both')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--latent-dim', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--l1', type=float, default=1e-3)
    ap.add_argument('--decor', type=float, default=5e-3)
    ap.add_argument('--ortho', type=float, default=1e-3)
    ap.add_argument('--unit', type=float, default=1e-3)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    A = np.load(OUT/'activations.npy'); input_dim = A.shape[1]

    if args.mode in ('regular','both'):
        reg = SparseAutoencoderSAE(input_dim, args.latent_dim, args.hidden)
        reg = _train(reg, A, epochs=args.epochs, l1_coef=args.l1, device=args.device)
        torch.save(reg.state_dict(), OUT/'sae_regular.pt')
        (OUT/'sae_regular_config.json').write_text(json.dumps({'input_dim':input_dim,'latent_dim':args.latent_dim,'hidden':args.hidden}, indent=2))
        print('Saved regular SAE.')
    if args.mode in ('monosemantic','both'):
        mono = MonosemanticSAE(input_dim, args.latent_dim, args.hidden, topk=args.topk)
        mono = _train(mono, A, epochs=args.epochs, l1_coef=args.l1, decor=args.decor, ortho=args.ortho, unit=args.unit, device=args.device)
        torch.save(mono.state_dict(), OUT/'sae_mono.pt')
        (OUT/'sae_mono_config.json').write_text(json.dumps({'input_dim':input_dim,'latent_dim':args.latent_dim,'hidden':args.hidden,'topk':args.topk}, indent=2))
        print('Saved monosemantic SAE.')

if __name__ == '__main__':
    main()

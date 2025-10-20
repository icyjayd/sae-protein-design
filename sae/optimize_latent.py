
#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
OUT = Path('outputs')
def ensemble_predict(models, X):
    import numpy as np
    preds = np.stack([m.predict(X) for m in models], axis=1)
    return preds.mean(axis=1), preds.std(axis=1)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', choices=['regular','mono'], default='mono')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--pop', type=int, default=256)
    ap.add_argument('--elite', type=int, default=32)
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--risk', type=float, default=0.25)
    args = ap.parse_args()
    Z = np.load(OUT/f'sparse_codes_{args.suffix}.npy')
    y = np.load(OUT/'labels.npy') if (OUT/'labels.npy').exists() else np.random.normal(0,1, len(Z))
    models = []
    for i in range(5):
        rf = RandomForestRegressor(n_estimators=300, random_state=100+i); rf.fit(Z, y); models.append(rf)
    rng = np.random.RandomState(0); dim = Z.shape[1]
    pop = rng.normal(0, 1.0, size=(args.pop, dim)); best=None; history=[]
    for t in range(args.iters):
        mean, std = ensemble_predict(models, pop); score = mean - args.risk * std
        elite_idx = np.argsort(score)[-args.elite:]; elites = pop[elite_idx]
        if best is None or np.max(score) > best[1]: best = (elites[-1], float(np.max(score)))
        history.append(float(np.max(score)))
        new_pop = []
        for e in elites:
            new_pop.append(e)
            for _ in range((args.pop // args.elite) - 1):
                new_pop.append(e + rng.normal(0, args.sigma, size=dim))
        pop = np.stack(new_pop)[:args.pop]
    np.save(OUT/f'opt_best_latent_{args.suffix}.npy', best[0]); np.save(OUT/f'opt_history_{args.suffix}.npy', np.array(history))
    print('Saved best latent and history.')
if __name__ == '__main__':
    main()

"""MAIN EXPERIMENT - three-layer GCN on Cora (Planetoid split 140/500/1000).

Protocol (mirrors a reproducible-baseline report):
  * fixed public split, test nodes never used for training or checkpoint selection
  * 10 seeds; 200 epochs each; checkpoint = lowest validation cross-entropy
  * same network with A_hat = I (no graph) as baseline
"""
import json, sys, time, pathlib, warnings, numpy as np
warnings.filterwarnings("ignore")
from cora_data import load_cora
from gcn_numpy import train_gcn, train_mlp_baseline, accuracy, confusion_and_macro

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs"
SEEDS = [42, 123, 456, 7, 8, 9, 10, 11, 12, 13]
CFG = dict(hidden=(64, 32), epochs=200, lr=0.01, dropout=0.5, wd=5e-4, patience=10**6, select="val_loss")

A, X, y, tr, va, te = load_cora(); C = int(y.max()) + 1
n_par = lambda dims: sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(3))
dims = [X.shape[1], *CFG["hidden"], C]
print(f"Cora: nodes={A.shape[0]} undirected edges={int(A.nnz/2)} features={X.shape[1]} classes={C}")
print(f"split train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}  | trainable parameters = {n_par(dims):,}")
print("class counts:", np.bincount(y).tolist(), "| test class counts:", np.bincount(y[te], minlength=C).tolist())
print(f"homophily = {float(A.multiply(y[:, None] == y[None, :]).sum() / A.sum()):.3f}   mean degree = {A.sum()/A.shape[0]:.2f}")
print("config:", {k: v for k, v in CFG.items() if k != 'patience'}); print("-" * 78, flush=True)

res = {"gcn": [], "mlp": []}; cm = {}; keep = {}
for s in SEEDS:
    for name, fn in (("gcn", lambda: train_gcn(A, X, y, tr, va, te, seed=s, **CFG)),
                     ("mlp", lambda: train_mlp_baseline(X, y, tr, va, te, seed=s, **{k: v for k, v in CFG.items()}))):
        t = time.time(); model, fin, h = fn()
        M, met = confusion_and_macro(fin["P"], y, te, C)
        met.update(seed=s, best_epoch=h["best_epoch"], best_val_loss=float(min(h["val_loss"])),
                   val_acc=accuracy(fin["P"], y, va), train_acc=accuracy(fin["P"], y, tr),
                   test_loss=float(-np.mean(np.log(fin["P"][te, y[te]] + 1e-12))))
        res[name].append(met); cm[f"{name}_{s}"] = M.tolist()
        print(f"[{name.upper()}] seed {s:3d} | best epoch {h['best_epoch']:3d} | val loss {met['best_val_loss']:.4f} | "
              f"test loss {met['test_loss']:.4f} | test acc {met['acc']:.4f} | macro-F1 {met['macro_f1']:.4f} | {time.time()-t:.0f}s", flush=True)
        if name == "gcn" and s == SEEDS[0]:
            keep = dict(hist=h, P=fin["P"], H1=np.maximum(0, fin["Z"][0]), H2=np.maximum(0, fin["Z"][1]))
    
def agg(name, key):
    v = np.array([r[key] for r in res[name]]); return float(v.mean()), float(v.std(ddof=1))
print("-" * 78)
for name in ("mlp", "gcn"):
    print(name.upper(), " | ".join(f"{k}: {agg(name,k)[0]:.4f} ± {agg(name,k)[1]:.4f}" for k in
          ("acc", "macro_precision", "macro_recall", "macro_f1", "test_loss", "best_val_loss")))
print("\nSeed", SEEDS[0], "GCN confusion matrix (rows actual, cols predicted):\n", np.array(cm[f"gcn_{SEEDS[0]}"]))
print("GCN per-class F1 (seed %d):" % SEEDS[0], np.round(res["gcn"][0]["per_class_f1"], 3).tolist())
json.dump(dict(cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()}, seeds=SEEDS, res=res, cm=cm,
               hist={k: ([float(x) for x in v] if isinstance(v, list) else v) for k, v in keep["hist"].items()},
               n_params=n_par(dims)), open(OUT / "cora_results.json", "w"))
np.savez_compressed(OUT / "cora_seed42.npz", P=keep["P"], H1=keep["H1"], H2=keep["H2"], y=y, te=te, tr=tr)

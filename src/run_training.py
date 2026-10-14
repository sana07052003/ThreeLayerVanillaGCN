"""PART B - TRAINABLE three-layer GCN on the 60-node community graph.
Prints data summary, training curve, predictions, GCN-vs-MLP comparison over 10 seeds."""
import json, pathlib, numpy as np
OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs"
from graph_data import make_sbm_graph
from gcn_ops import sym_normalize
from gcn_numpy import train_gcn, train_mlp_baseline, accuracy

np.set_printoptions(precision=3, suppress=True, linewidth=120)
A, X, y, tr, va, te = make_sbm_graph(seed=0)
print(f"nodes={len(y)}  undirected edges={int(A.sum()/2)}  features={X.shape[1]}  classes={y.max()+1}")
print(f"train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}   mean degree={A.sum(1).mean():.2f}")
print("homophily (fraction of edges joining same-class nodes) = %.3f" %
      ((A * (y[:, None] == y[None, :])).sum() / A.sum()))

model, fin, hist = train_gcn(A, X, y, tr, va, te, seed=0, verbose=True)
P = fin["P"]
res = {k: accuracy(P, y, m) for k, m in (("train", tr), ("val", va), ("test", te))}
print("\nGCN (seed 0) accuracy:", res)
print("epochs run:", len(hist["train_loss"]), " final train loss: %.4f" % hist["train_loss"][-1])
print("\nFirst 8 test nodes: true / predicted / softmax")
for i in np.where(te)[0][:8]:
    print(f"  node {i:2d}  true={y[i]}  pred={P[i].argmax()}  P={P[i]}")
print("\nLayer-wise embedding shapes:", [c.shape for c in fin["Z"]])

# multi-seed comparison (model-init seed AND data seed vary)
g, m = [], []
for s in range(10):
    A_, X_, y_, tr_, va_, te_ = make_sbm_graph(seed=s)
    _, f1, _ = train_gcn(A_, X_, y_, tr_, va_, te_, seed=s)
    _, f2, _ = train_mlp_baseline(X_, y_, tr_, va_, te_, seed=s)
    g.append(accuracy(f1["P"], y_, te_)); m.append(accuracy(f2["P"], y_, te_))
print("\n10-run test accuracy (different graphs/splits/inits)")
print("  3-layer GCN : %.3f +/- %.3f   runs=%s" % (np.mean(g), np.std(g), np.round(g, 2)))
print("  3-layer MLP : %.3f +/- %.3f   runs=%s" % (np.mean(m), np.std(m), np.round(m, 2)))

np.savez(str(OUT / "results.npz"), P=P, y=y, A=A, X=X, tr=tr, va=va, te=te,
         H1=np.maximum(0, fin["Z"][0]), H2=np.maximum(0, fin["Z"][1]), Z3=fin["Z"][2])
json.dump(dict(single_run=res, hist={k: ([float(v) for v in vals] if isinstance(vals, list) else vals) for k, vals in hist.items()},
               gcn_runs=g, mlp_runs=m, n_nodes=int(len(y)), n_edges=int(A.sum() / 2)),
          open(str(OUT / "results.json"), "w"))

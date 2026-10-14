"""Finite-difference check of the hand-derived back-propagation."""
import sys, numpy as np
sys.path.insert(0, "src")
from graph_data import make_sbm_graph
from gcn_ops import sym_normalize
from gcn_numpy import GCN3

A, X, y, tr, va, te = make_sbm_graph(seed=0)
Ah = sym_normalize(A)
m = GCN3([8, 16, 8, 3], seed=1, dropout=0.0, weight_decay=5e-4)
c = m.forward(Ah, X); gW, gb = m.backward(c, y, tr)
rng = np.random.default_rng(0); worst = 0.0
for name, params, grads in (("W", m.W, gW), ("b", m.b, gb)):
    for l in range(3):
        for _ in range(5):
            i = tuple(rng.integers(0, s) for s in params[l].shape)
            old = params[l][i]; h = 1e-5
            params[l][i] = old + h; lp = m.loss(m.forward(Ah, X), y, tr)
            params[l][i] = old - h; lm = m.loss(m.forward(Ah, X), y, tr)
            params[l][i] = old
            num = (lp - lm) / (2 * h); ana = grads[l][i]
            rel = abs(num - ana) / max(1e-10, abs(num) + abs(ana))
            worst = max(worst, rel)
            print(f"{name}{l+1}{[int(k) for k in i]}  analytic={ana: .6e}  numeric={num: .6e}  rel.err={rel:.2e}")
print(f"\nWorst relative error over 30 checked entries: {worst:.2e}")
print("GRADIENT CHECK", "PASSED" if worst < 1e-5 else "FAILED")

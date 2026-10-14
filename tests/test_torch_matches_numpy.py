"""Checks that the PyTorch layers reproduce the NumPy forward pass and gradients
when given identical weights (run locally: requires torch)."""
import sys, numpy as np, torch
sys.path.insert(0, "src")
from graph_data import make_sbm_graph
from gcn_ops import sym_normalize
import gcn_numpy as gn, gcn_torch as gt

A, X, y, tr, va, te = make_sbm_graph(seed=0)
Ah = sym_normalize(A)
np_model = gn.GCN3([8, 16, 8, 3], seed=1, dropout=0.0, weight_decay=0.0)
cache = np_model.forward(Ah, X); gW, gb = np_model.backward(cache, y, tr)

t_model = gt.GCN3(8, 16, 8, 3, dropout=0.0)
for l, layer in enumerate((t_model.l1, t_model.l2, t_model.l3)):
    layer.W.data = torch.tensor(np_model.W[l]); layer.b.data = torch.tensor(np_model.b[l])
At = torch.tensor(Ah); Xt = torch.tensor(X); yt = torch.tensor(y); trt = torch.tensor(tr)
out = t_model(At, Xt.double())
loss = torch.nn.functional.cross_entropy(out[trt], yt[trt]); loss.backward()
assert np.allclose(torch.softmax(out, 1).detach().numpy(), cache["P"], atol=1e-8), "forward mismatch"
for l, layer in enumerate((t_model.l1, t_model.l2, t_model.l3)):
    assert np.allclose(layer.W.grad.numpy(), gW[l], atol=1e-8), f"dW{l+1} mismatch"
    assert np.allclose(layer.b.grad.numpy(), gb[l], atol=1e-8), f"db{l+1} mismatch"
print("PyTorch and NumPy implementations agree (forward pass + all gradients).")

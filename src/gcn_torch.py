"""PyTorch version of the SAME three-layer GCN (autograd instead of hand-derived gradients).

NOTE: this file could not be executed in the environment where the assignment
package was produced (no PyTorch install possible there). All numbers reported
in the README/report come from the NumPy implementation. Run
`python tests/test_torch_matches_numpy.py` locally to verify the two agree.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adjacency(A: torch.Tensor) -> torch.Tensor:
    A_t = A + torch.eye(A.shape[0], dtype=A.dtype)          # self loops
    d = A_t.sum(1)
    d_inv_sqrt = d.pow(-0.5)
    return d_inv_sqrt[:, None] * A_t * d_inv_sqrt[None, :]   # D^-1/2 A~ D^-1/2


class GraphConv(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.W = nn.Parameter(torch.empty(fin, fout))
        self.b = nn.Parameter(torch.zeros(1, fout))
        nn.init.xavier_uniform_(self.W)

    def forward(self, A_hat, H):
        return A_hat @ H @ self.W + self.b                   # A_hat H W + b


class GCN3(nn.Module):
    def __init__(self, fin, h1, h2, n_classes, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.l3 = GraphConv(fin, h1), GraphConv(h1, h2), GraphConv(h2, n_classes)
        self.p = dropout

    def forward(self, A_hat, X):
        H = F.relu(self.l1(A_hat, F.dropout(X, self.p, self.training)))
        H = F.relu(self.l2(A_hat, F.dropout(H, self.p, self.training)))
        return self.l3(A_hat, F.dropout(H, self.p, self.training))   # logits


def train(A, X, y, tr, va, te, hidden=(16, 8), epochs=300, lr=0.02, wd=5e-4, dropout=0.5, seed=0,
          patience=60, select="val_acc"):
    """A may be a dense numpy array (use A.toarray() for scipy-sparse Cora; 2708^2 floats is fine)."""
    torch.manual_seed(seed)
    A, X = torch.tensor(A, dtype=torch.float32), torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y); tr, va, te = (torch.tensor(m) for m in (tr, va, te))
    A_hat = normalized_adjacency(A)
    model = GCN3(X.shape[1], hidden[0], hidden[1], int(y.max()) + 1, dropout)
    # weight decay only on first layer, as in the NumPy version
    opt = torch.optim.Adam([{"params": model.l1.parameters(), "weight_decay": wd},
                            {"params": list(model.l2.parameters()) + list(model.l3.parameters()),
                             "weight_decay": 0.0}], lr=lr)
    best, best_state, bad = -float("inf"), None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(A_hat, X)[tr], y[tr]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(A_hat, X)
            vacc = (out[va].argmax(1) == y[va]).float().mean().item()
            vloss = F.cross_entropy(out[va], y[va]).item()
        score = -vloss if select == "val_loss" else vacc - 1e-3 * vloss
        if score > best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        out = model(A_hat, X)
    acc = lambda m: (out[m].argmax(1) == y[m]).float().mean().item()
    return model, {"train": acc(tr), "val": acc(va), "test": acc(te)}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from cora_data import load_cora
    A, X, y, tr, va, te = load_cora()
    _, res = train(A.toarray(), X, y, tr, va, te, hidden=(64, 32), epochs=200, lr=0.01,
                   seed=42, patience=10**6, select="val_loss")
    print("PyTorch GCN accuracy on Cora (seed 42):", res)

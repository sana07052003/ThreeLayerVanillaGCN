"""Trainable three-layer GCN in pure NumPy with hand-derived back-propagation.

    H1 = ReLU(A_hat X  W1 + b1)
    H2 = ReLU(A_hat H1 W2 + b2)
    Z  =      A_hat H2 W3 + b3 ,   P = softmax(Z)

Optimiser: Adam (implemented from scratch). Init: Xavier/Glorot uniform.
Regularisation: L2 weight decay on W1 and inverted dropout on layer inputs.
"""
import numpy as np
import scipy.sparse as sp
from gcn_ops import sym_normalize, relu, softmax, cross_entropy


class GCN3:
    def __init__(self, dims, seed=0, dropout=0.5, weight_decay=5e-4):
        assert len(dims) == 4, "exactly three graph-convolution layers"
        self.rng = np.random.default_rng(seed)
        self.dropout, self.wd = dropout, weight_decay
        self.W, self.b = [], []
        for i in range(3):
            lim = np.sqrt(6.0 / (dims[i] + dims[i + 1]))        # Xavier uniform
            self.W.append(self.rng.uniform(-lim, lim, (dims[i], dims[i + 1])))
            self.b.append(np.zeros((1, dims[i + 1])))
        self._adam_init()

    # ---------- forward ----------
    def forward(self, A_hat, X, train=False):
        cache = {"A": A_hat, "inp": [], "AH": [], "Z": [], "mask": []}
        H = X
        for l in range(3):
            if train and self.dropout > 0:                      # inverted dropout
                m = (self.rng.random(H.shape) > self.dropout) / (1 - self.dropout)
                H = H * m
            else:
                m = None
            AH = A_hat @ H
            Z = AH @ self.W[l] + self.b[l]
            cache["inp"].append(H); cache["AH"].append(AH)
            cache["Z"].append(Z); cache["mask"].append(m)
            H = relu(Z) if l < 2 else Z                         # last layer: logits
        cache["P"] = softmax(H)
        return cache

    # ---------- backward ----------
    def backward(self, cache, y, train_mask):
        A, P = cache["A"], cache["P"]
        n_tr = train_mask.sum()
        Y = np.zeros_like(P); Y[np.arange(len(y)), y] = 1.0
        dZ = (P - Y) * train_mask[:, None] / n_tr               # dL/dZ3 (softmax+CE)
        gW, gb = [None] * 3, [None] * 3
        for l in (2, 1, 0):
            gW[l] = cache["AH"][l].T @ dZ                       # dL/dW_l
            gb[l] = dZ.sum(axis=0, keepdims=True)               # dL/db_l
            if l > 0:
                dH = A.T @ (dZ @ self.W[l].T)                   # back through A_hat and W
                if cache["mask"][l] is not None:
                    dH = dH * cache["mask"][l]                  # dropout mask
                dZ = dH * (cache["Z"][l - 1] > 0)               # ReLU derivative
        gW[0] = gW[0] + self.wd * self.W[0]                     # L2 on first layer
        return gW, gb

    def loss(self, cache, y, mask):
        return cross_entropy(cache["P"], y, mask) + 0.5 * self.wd * np.sum(self.W[0] ** 2)

    # ---------- Adam ----------
    def _adam_init(self):
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def step(self, gW, gb, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params, grads = self.W + self.b, gW + gb
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)                  # in-place update

    def get_state(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_state(self, s):
        self.W, self.b = [w.copy() for w in s[0]], [b.copy() for b in s[1]]


def accuracy(P, y, mask):
    return float((P.argmax(1)[mask] == y[mask]).mean())


def confusion_and_macro(P, y, mask, C):
    """Confusion matrix (rows actual, cols predicted) and macro precision/recall/F1.
    Undefined divisions are set to 0, as in the reference report."""
    pred = P.argmax(1)[mask]; act = y[mask]
    M = np.zeros((C, C), int)
    for a, p in zip(act, pred):
        M[a, p] += 1
    tp = np.diag(M).astype(float)
    prec = np.divide(tp, M.sum(0), out=np.zeros(C), where=M.sum(0) > 0)
    rec = np.divide(tp, M.sum(1), out=np.zeros(C), where=M.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(C), where=(prec + rec) > 0)
    return M, dict(acc=float(tp.sum() / M.sum()), macro_precision=float(prec.mean()),
                   macro_recall=float(rec.mean()), macro_f1=float(f1.mean()), per_class_f1=f1.tolist())


def train_gcn(A, X, y, tr, va, te, hidden=(16, 8), epochs=300, lr=0.02,
              dropout=0.5, wd=5e-4, seed=0, patience=60, verbose=False, select="val_acc"):
    """select: 'val_acc' (val accuracy, loss as tie-break) or 'val_loss' (lowest validation
    cross-entropy checkpoint, the criterion used in the reference report)."""
    A_hat = sym_normalize(A)
    C = int(y.max()) + 1
    model = GCN3([X.shape[1], hidden[0], hidden[1], C], seed, dropout, wd)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, bad, best_ep = -np.inf, model.get_state(), 0, 1
    for ep in range(epochs):
        c = model.forward(A_hat, X, train=True)
        gW, gb = model.backward(c, y, tr)
        model.step(gW, gb, lr=lr)
        e = model.forward(A_hat, X, train=False)               # evaluation (no dropout)
        hist["train_loss"].append(cross_entropy(e["P"], y, tr))
        hist["val_loss"].append(cross_entropy(e["P"], y, va))
        hist["train_acc"].append(accuracy(e["P"], y, tr))
        hist["val_acc"].append(accuracy(e["P"], y, va))
        score = (-hist["val_loss"][-1]) if select == "val_loss" else \
                (hist["val_acc"][-1] - 1e-3 * hist["val_loss"][-1])
        if score > best or ep == 0:
            best, best_state, bad, best_ep = score, model.get_state(), 0, ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and (ep % 25 == 0):
            print(f"epoch {ep:3d}  loss {hist['train_loss'][-1]:.4f}  "
                  f"val_acc {hist['val_acc'][-1]:.3f}")
    model.set_state(best_state)
    final = model.forward(A_hat, X, train=False)
    hist["best_epoch"] = best_ep
    return model, final, hist


def train_mlp_baseline(X, y, tr, va, te, hidden=(16, 8), epochs=300, lr=0.02, seed=0, **kw):
    """Same 3-layer network but A_hat replaced by I (no graph) -> isolates the
    contribution of neighbourhood aggregation."""
    n = X.shape[0]
    return train_gcn(sp.csr_matrix((n, n)), X, y, tr, va, te, hidden, epochs, lr, seed=seed, **kw)

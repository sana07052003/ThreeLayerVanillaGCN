"""Two-layer vanilla GCN in NumPy with hand-derived back-propagation.

    H1 = ReLU(A_hat X  W1 + b1)
    Z  =      A_hat H1 W2 + b2 ,    P = softmax(Z)

Same building blocks (gcn_ops), optimiser (Adam) and checkpoint rule as the
three-layer model, so the two depths can be compared fairly.
"""
import numpy as np
from gcn_ops import sym_normalize, relu, softmax, cross_entropy
from gcn_numpy import accuracy


class GCN2:
    def __init__(self, dims, seed=0, dropout=0.5, weight_decay=5e-4):
        assert len(dims) == 3, "exactly two graph-convolution layers"
        self.rng = np.random.default_rng(seed)
        self.dropout, self.wd = dropout, weight_decay
        self.W, self.b = [], []
        for i in range(2):
            lim = np.sqrt(6.0 / (dims[i] + dims[i + 1]))          # Xavier uniform
            self.W.append(self.rng.uniform(-lim, lim, (dims[i], dims[i + 1])))
            self.b.append(np.zeros((1, dims[i + 1])))
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, A_hat, X, train=False):
        cache = {"A": A_hat, "AH": [], "Z": [], "mask": []}
        H = X
        for l in range(2):
            if train and self.dropout > 0:
                m = (self.rng.random(H.shape) > self.dropout) / (1 - self.dropout)
                H = H * m
            else:
                m = None
            AH = A_hat @ H                                        # aggregation
            Z = AH @ self.W[l] + self.b[l]                        # transform
            cache["AH"].append(AH); cache["Z"].append(Z); cache["mask"].append(m)
            H = relu(Z) if l == 0 else Z                          # ReLU only after layer 1
        cache["P"] = softmax(H)
        return cache

    def backward(self, cache, y, train_mask):
        A, P = cache["A"], cache["P"]
        Y = np.zeros_like(P); Y[np.arange(len(y)), y] = 1.0
        dZ = (P - Y) * train_mask[:, None] / train_mask.sum()     # softmax + cross-entropy
        gW, gb = [None] * 2, [None] * 2
        gW[1] = cache["AH"][1].T @ dZ; gb[1] = dZ.sum(0, keepdims=True)
        dH = A.T @ (dZ @ self.W[1].T)                             # back through layer 2
        if cache["mask"][1] is not None:
            dH = dH * cache["mask"][1]
        dZ1 = dH * (cache["Z"][0] > 0)                            # ReLU derivative
        gW[0] = cache["AH"][0].T @ dZ1 + self.wd * self.W[0]
        gb[0] = dZ1.sum(0, keepdims=True)
        return gW, gb

    def loss(self, cache, y, mask):
        return cross_entropy(cache["P"], y, mask) + 0.5 * self.wd * np.sum(self.W[0] ** 2)

    def step(self, gW, gb, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):  # Adam
        self.t += 1
        for k, (p, g) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            p -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)

    def get_state(self): return [w.copy() for w in self.W], [b.copy() for b in self.b]
    def set_state(self, s): self.W, self.b = [w.copy() for w in s[0]], [b.copy() for b in s[1]]


def train_two_layer(A, X, y, tr, va, hidden=64, epochs=200, lr=0.01, dropout=0.5, wd=5e-4,
                    seed=0, select="val_loss", log_every=0):
    A_hat = sym_normalize(A); C = int(y.max()) + 1
    model = GCN2([X.shape[1], hidden, C], seed, dropout, wd)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, best_ep = -np.inf, model.get_state(), 1
    for ep in range(epochs):
        c = model.forward(A_hat, X, train=True)
        gW, gb = model.backward(c, y, tr); model.step(gW, gb, lr=lr)
        e = model.forward(A_hat, X, train=False)
        hist["train_loss"].append(cross_entropy(e["P"], y, tr)); hist["val_loss"].append(cross_entropy(e["P"], y, va))
        hist["train_acc"].append(accuracy(e["P"], y, tr)); hist["val_acc"].append(accuracy(e["P"], y, va))
        if select == "last":                                      # no validation set available: keep final epoch
            score = float(ep)
        else:
            score = -hist["val_loss"][-1] if select == "val_loss" else hist["val_acc"][-1] - 1e-3 * hist["val_loss"][-1]
        if score > best or ep == 0:
            best, best_state, best_ep = score, model.get_state(), ep + 1
        if log_every and ep % log_every == 0:
            print(f"Epoch {ep:4d} | train loss {hist['train_loss'][-1]:.4f} | train acc {hist['train_acc'][-1]:.3f} | "
                  f"eval-set acc {hist['val_acc'][-1]:.3f}")
    model.set_state(best_state)
    hist["best_epoch"] = best_ep
    return model, model.forward(A_hat, X, train=False), hist

"""Datasets used in the assignment.

1. toy_graph()      : 4-node graph for the *fixed-weight* hand-checkable demo.
2. make_sbm_graph() : 60-node, 3-community graph (stochastic block model) with
                      noisy node features, for the *trainable* GCN.
Both are generated deterministically from a seed -> fully reproducible.
"""
import numpy as np


def toy_graph():
    """Nodes 0..3, edges: 0-1, 1-2, 1-3 (a small 'star' around node 1)."""
    edges = [(0, 1), (1, 2), (1, 3)]
    A = np.zeros((4, 4))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    X = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0],
                  [2.0, 0.0]])
    return A, X, edges


def make_sbm_graph(n_per_class=20, n_classes=3, n_features=8,
                   p_in=0.30, p_out=0.02, noise=1.6, seed=0):
    """Stochastic-block-model graph. Node features are only weakly informative
    (class signal buried in Gaussian noise), so using the graph helps."""
    rng = np.random.default_rng(seed)
    n = n_per_class * n_classes
    y = np.repeat(np.arange(n_classes), n_per_class)

    P = np.where(y[:, None] == y[None, :], p_in, p_out)
    U = rng.random((n, n))
    A = np.triu((U < P).astype(float), k=1)
    A = A + A.T

    protos = rng.normal(0, 1, (n_classes, n_features))
    X = protos[y] + noise * rng.normal(0, 1, (n, n_features))

    # semi-supervised split: 4 labelled / 4 validation per class, rest test
    train = np.zeros(n, bool); val = np.zeros(n, bool); test = np.zeros(n, bool)
    for c in range(n_classes):
        idx = rng.permutation(np.where(y == c)[0])
        train[idx[:4]] = True
        val[idx[4:8]] = True
        test[idx[8:]] = True
    return A, X, y, train, val, test

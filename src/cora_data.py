"""Loader for the Cora citation network in the standard Planetoid format
(Yang et al., 2016): 2,708 papers, 5,278 unique undirected citation links, 1,433 binary bag-of-words
features, 7 subject classes, and the fixed split 140 train / 500 validation / 1,000 test.
Files live in data/cora/ (downloaded from github.com/kimiyoung/planetoid)."""
import pickle, pathlib, numpy as np, scipy.sparse as sp

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "cora"


def _load(name):
    with open(DATA / f"ind.cora.{name}", "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_cora(row_normalize=True):
    x, y_, tx, ty, allx, ally, graph = (_load(n) for n in ("x", "y", "tx", "ty", "allx", "ally", "graph"))
    test_idx = np.loadtxt(DATA / "ind.cora.test.index", dtype=int)
    test_sorted = np.sort(test_idx)
    X = sp.vstack((allx, tx)).tolil(); Y = np.vstack((ally, ty))
    X[test_idx, :] = X[test_sorted, :]; Y[test_idx, :] = Y[test_sorted, :]
    n = X.shape[0]
    rows, cols = [], []
    for i, nbrs in graph.items():
        for j in nbrs:
            rows.append(i); cols.append(j)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float); A.setdiag(0); A.eliminate_zeros()      # undirected, no loops
    X = X.tocsr().astype(float)
    if row_normalize:                                                         # standard preprocessing
        r = np.asarray(X.sum(1)).ravel(); r[r == 0] = 1; X = sp.diags(1 / r) @ X
    y = Y.argmax(1)
    tr = np.zeros(n, bool); va = np.zeros(n, bool); te = np.zeros(n, bool)
    n_tr = y_.shape[0]                                    # 140 labelled training nodes
    tr[:n_tr] = True; va[n_tr:n_tr + 500] = True; te[test_idx] = True
    return A, np.asarray(X.todense()), y, tr, va, te

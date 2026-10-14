"""Mathematical building blocks of the Vanilla GCN (NumPy).
Each function corresponds to one formula of Phase 2."""
import numpy as np
import scipy.sparse as sp


def add_self_loops(A):
    """A_tilde = A + I                                   (Eq. 2)"""
    return A + np.eye(A.shape[0])


def degree_matrix(A_tilde):
    """D_tilde_ii = sum_j A_tilde_ij                      (Eq. 3)"""
    return np.diag(A_tilde.sum(axis=1))


def sym_normalize(A):
    """A_hat = D^{-1/2} (A + I) D^{-1/2}                  (Eq. 4)"""
    if sp.issparse(A):                                   # sparse path (e.g. Cora)
        At = A + sp.identity(A.shape[0], format="csr")
        d = np.asarray(At.sum(axis=1)).ravel()
        Dm = sp.diags(1.0 / np.sqrt(d))
        return (Dm @ At @ Dm).tocsr()
    At = add_self_loops(A)
    d = At.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    return d_inv_sqrt[:, None] * At * d_inv_sqrt[None, :]


def relu(Z):
    """ReLU(z) = max(0, z)                                (Eq. 6)"""
    return np.maximum(0.0, Z)


def softmax(Z):
    """softmax row-wise, numerically stable                (Eq. 10)"""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def gcn_layer(A_hat, H, W, b=None, activation=relu):
    """H' = sigma(A_hat H W + b)                          (Eq. 5)
    Returns (pre_activation, output, aggregated A_hat H) for inspection."""
    AH = A_hat @ H                  # 1) neighbourhood aggregation
    Z = AH @ W                      # 2) linear feature transform
    if b is not None:
        Z = Z + b
    out = activation(Z) if activation is not None else Z   # 3) non-linearity
    return Z, out, AH


def cross_entropy(P, y, mask):
    """Masked mean negative log-likelihood                  (Eq. 11)"""
    idx = np.where(mask)[0]
    return -np.mean(np.log(P[idx, y[idx]] + 1e-12))

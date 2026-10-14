"""Figures for the Cora experiment (run after run_cora.py)."""
import json, pathlib, warnings, numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from cora_data import load_cora

ROOT = pathlib.Path(__file__).resolve().parent.parent; FIG = ROOT / "outputs" / "figures"
CR = json.load(open(ROOT / "outputs" / "cora_results.json")); D = np.load(ROOT / "outputs" / "cora_seed42.npz")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
B, O = "#2a6fdb", "#e8743b"

h = CR["hist"]; ep = np.arange(1, len(h["train_loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
ax[0].plot(ep, h["train_loss"], color=B, label="train loss"); ax[0].plot(ep, h["val_loss"], color=O, label="validation loss")
ax[0].axvline(h["best_epoch"], color="k", ls=":", lw=1); ax[0].text(h["best_epoch"] + 2, max(h["train_loss"]) * .8, f"selected\nepoch {h['best_epoch']}", fontsize=8)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend(frameon=False); ax[0].set_title("Loss (seed 42)")
ax[1].plot(ep, h["train_acc"], color=B, label="train acc (140 nodes)"); ax[1].plot(ep, h["val_acc"], color=O, label="validation acc (500 nodes)")
ax[1].axvline(h["best_epoch"], color="k", ls=":", lw=1); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(frameon=False, loc="lower right"); ax[1].set_title("Accuracy (seed 42)")
plt.tight_layout(); plt.savefig(FIG / "cora_curves.png", dpi=170); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a, s in zip(ax, CR["seeds"][:3]):
    M = np.array(CR["cm"][f"gcn_{s}"]); a.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    for i in range(7):
        for j in range(7): a.text(j, i, M[i, j], ha="center", va="center", fontsize=8, color="white" if M[i, j] > M.max() * .55 else "black")
    a.set_title(f"GCN, seed {s}"); a.set_xlabel("predicted class"); a.set_ylabel("actual class"); a.set_xticks(range(7)); a.set_yticks(range(7))
plt.tight_layout(); plt.savefig(FIG / "cora_confusion.png", dpi=170); plt.close()

g = np.array([r["acc"] for r in CR["res"]["gcn"]]); m = np.array([r["acc"] for r in CR["res"]["mlp"]])
fig, ax = plt.subplots(figsize=(8.2, 3.8)); x = np.arange(len(g)); w = .38
ax.bar(x - w / 2, m, w, color=O, label=f"no graph (Â = I)  mean {m.mean():.3f}"); ax.bar(x + w / 2, g, w, color=B, label=f"3-layer GCN  mean {g.mean():.3f}")
ax.set_xticks(x); ax.set_xticklabels(CR["seeds"]); ax.set_xlabel("seed"); ax.set_ylabel("test accuracy (1000 nodes)"); ax.set_ylim(0, 1); ax.legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "cora_gcn_vs_mlp.png", dpi=170); plt.close()

A, X, y, tr, va, te = load_cora()
Xp = PCA(50, random_state=0).fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2)); cmap = plt.get_cmap("tab10")
for a, Z, t in ((ax[0], Xp, "Input features (PCA-50 → t-SNE)"), (ax[1], D["H2"], "GCN layer-2 embedding H² (32-d → t-SNE)")):
    E = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(Z)
    for c in range(7): a.scatter(*E[y == c].T, s=5, color=cmap(c), label=f"class {c}")
    a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([]); a.spines[["left", "bottom"]].set_visible(False)
ax[1].legend(markerscale=3, frameon=False, fontsize=8, loc="upper right", ncol=2)
plt.tight_layout(); plt.savefig(FIG / "cora_tsne.png", dpi=170); plt.close(); print("done")

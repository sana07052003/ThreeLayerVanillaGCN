"""Figures for the two-layer chapter (run after run_two_layer_example.py)."""
import json, pathlib, numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parent.parent; FIG = ROOT / "outputs" / "figures"
D = json.load(open(ROOT / "outputs" / "two_layer_example.json"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
B, O = "#2a6fdb", "#e8743b"

# --- architecture (two layers)
fig, ax = plt.subplots(figsize=(12, 3.6)); ax.axis("off"); ax.set_xlim(0, 20); ax.set_ylim(0, 5.4)
def box(x, w, txt, fc, ec):
    ax.add_patch(FancyBboxPatch((x, 1.5), w, 2.4, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.4)); ax.text(x + w / 2, 2.7, txt, ha="center", va="center", fontsize=9.5)
def arr(a, b): ax.add_patch(FancyArrowPatch((a, 2.7), (b, 2.7), arrowstyle="-|>", mutation_scale=14, color="#444"))
box(0.2, 2.6, "Input\n$X$\n$N\\times F$", "#cfe0ff", "#2a6fdb"); arr(2.9, 3.7)
box(3.7, 5.2, "GCN Layer 1\n$H^{(1)}=\\mathrm{ReLU}(\\hat{A}XW^{(1)}+b^{(1)})$\n$W^{(1)}: F\\times h$", "#ffe3c9", "#e8743b"); arr(9.0, 9.8)
box(9.8, 5.2, "GCN Layer 2\n$Z=\\hat{A}H^{(1)}W^{(2)}+b^{(2)}$\n$W^{(2)}: h\\times C$ (no ReLU)", "#d8f1ea", "#2ca58d"); arr(15.1, 15.9)
box(15.9, 3.9, "softmax → $P$\n$N\\times C$\nloss on labelled\nnodes only", "#eee", "#555")
ax.text(10, 4.9, "$\\hat{A}=\\tilde{D}^{-1/2}(A+I)\\tilde{D}^{-1/2}$ is shared by both layers (computed once) · receptive field = 2 hops", ha="center", fontsize=10.5)
plt.savefig(FIG / "two_layer_architecture.png", dpi=170, bbox_inches="tight"); plt.close()

# --- karate graph: truth vs prediction
A = np.array(D["A"]); y = np.array(D["y"]); P = np.array(D["P"]); tr = np.array(D["tr"]); pred = P.argmax(1)
G = nx.from_numpy_array(A); pos = nx.spring_layout(G, seed=7, k=0.5); COL = [B, O]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for a, lab, t in ((ax[0], y, "True factions (Mr. Hi = blue, Officer = orange)"), (ax[1], pred, "Two-layer GCN prediction (final epoch)")):
    nx.draw_networkx_edges(G, pos, ax=a, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=[COL[v] for v in lab], node_size=260, ax=a, edgecolors="white")
    nx.draw_networkx_labels(G, pos, ax=a, font_size=7, font_color="white"); a.set_title(t, fontsize=10); a.axis("off")
lab_nodes = [i for i in range(34) if tr[i]]
nx.draw_networkx_nodes(G, pos, nodelist=lab_nodes, node_shape="s", node_color="none", edgecolors="black", node_size=520, linewidths=2, ax=ax[1])
wrong = [i for i in range(34) if pred[i] != y[i]]
nx.draw_networkx_nodes(G, pos, nodelist=wrong, node_shape="o", node_color="none", edgecolors="red", node_size=520, linewidths=2.2, ax=ax[1])
ax[1].text(0.02, 0.02, "black square = labelled node · red ring = misclassified", transform=ax[1].transAxes, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "two_layer_karate_graph.png", dpi=170); plt.close()

# --- curves
h = D["hist"]; ep = np.arange(1, len(h["train_loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))
ax[0].plot(ep, h["train_loss"], color=B); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy on the 2 labelled nodes"); ax[0].set_title("Training loss")
ax[1].plot(ep, h["val_acc"], color=O); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy on the 32 unlabelled nodes"); ax[1].set_ylim(0.5, 1.02)
ax[1].set_title("Accuracy on unlabelled nodes (monitored only)")
plt.tight_layout(); plt.savefig(FIG / "two_layer_curves.png", dpi=170); plt.close()

# --- node probability bar chart
fig, ax = plt.subplots(figsize=(11, 3.4)); idx = np.arange(34)
ax.bar(idx, P[:, 1], color=[COL[v] for v in y]); ax.axhline(0.5, color="k", ls="--", lw=1)
ax.set_xticks(idx); ax.set_xticklabels(idx, fontsize=7); ax.set_xlabel("node"); ax.set_ylabel("P(Officer)")
ax.set_title("Predicted probability of the 'Officer' faction; bar colour = true faction")
plt.tight_layout(); plt.savefig(FIG / "two_layer_probs.png", dpi=170); plt.close(); print("ok")

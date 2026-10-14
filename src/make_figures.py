"""Generate all figures from the saved outputs (run after run_training.py)."""
import json, pathlib, numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from graph_data import toy_graph
from gcn_ops import sym_normalize, add_self_loops

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"; FIG.mkdir(exist_ok=True, parents=True)
COL = ["#2a6fdb", "#e8743b", "#2ca58d"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig 1: toy graph + matrices
A, X, edges = toy_graph(); Ah = sym_normalize(A)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.1, 1, 1]})
G = nx.from_numpy_array(A); pos = {0: (0, 0), 1: (1, 0.5), 2: (2, 1), 3: (2, 0)}
nx.draw_networkx(G, pos, ax=ax[0], node_color="#cfe0ff", node_size=900, edgecolors="#2a6fdb",
                 font_weight="bold")
for i, (x, yv) in pos.items():
    ax[0].text(x, yv - 0.22, f"x={X[i].astype(int).tolist()}", ha="center", fontsize=8)
ax[0].set_title("Toy graph: 4 nodes, 3 edges\n(node features x_i shown below)"); ax[0].axis("off")
ax[0].set_ylim(-0.5, 1.3)
for a, M, t in ((ax[1], add_self_loops(A), r"$\tilde{A}=A+I$"), (ax[2], Ah, r"$\hat{A}=\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$")):
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=1.0); a.set_title(t)
    a.set_xticks(range(4)); a.set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{M[i,j]:.2f}" if M[i, j] % 1 else f"{int(M[i,j])}", ha="center", va="center",
                   color="white" if M[i, j] > 0.55 else "black", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "fig1_toy_graph_matrices.png", dpi=170); plt.close()

# ---- Fig 2: message passing illustration for node 1
fig, ax = plt.subplots(figsize=(7.5, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4)
nodes = {"x0": (1.2, 3.2), "x2": (1.2, 2.0), "x3": (1.2, 0.8), "x1 (self-loop)": (1.2, -0.0)}
pts = [("x_0", 3.4), ("x_2", 2.4), ("x_3", 1.4), ("x_1 (self)", 0.4)]
w = [Ah[1, 0], Ah[1, 2], Ah[1, 3], Ah[1, 1]]
for (lab, yv), wi in zip(pts, w):
    ax.add_patch(FancyBboxPatch((0.3, yv - 0.3), 2.2, 0.6, boxstyle="round,pad=0.05", fc="#cfe0ff", ec="#2a6fdb"))
    ax.text(1.4, yv, lab, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((2.6, yv), (5.0, 2.0), arrowstyle="-|>", mutation_scale=14, color="#555"))
    ax.text(3.7, (yv + 2.0) / 2 + 0.12, f"×{wi:.3f}", fontsize=9, color="#c0392b")
ax.add_patch(FancyBboxPatch((5.0, 1.55), 1.4, 0.9, boxstyle="round,pad=0.05", fc="#ffe3c9", ec="#e8743b"))
ax.text(5.7, 2.0, "Σ  (sum)", ha="center", va="center")
ax.add_patch(FancyArrowPatch((6.5, 2.0), (7.2, 2.0), arrowstyle="-|>", mutation_scale=14))
ax.add_patch(FancyBboxPatch((7.2, 1.55), 1.0, 0.9, boxstyle="round,pad=0.05", fc="#d8f1ea", ec="#2ca58d"))
ax.text(7.7, 2.0, "·W", ha="center", va="center")
ax.add_patch(FancyArrowPatch((8.3, 2.0), (8.8, 2.0), arrowstyle="-|>", mutation_scale=14))
ax.add_patch(FancyBboxPatch((8.8, 1.55), 1.1, 0.9, boxstyle="round,pad=0.05", fc="#eee", ec="#555"))
ax.text(9.35, 2.0, "ReLU", ha="center", va="center")
ax.set_title("Message passing at node 1 (weights = entries of row 1 of $\\hat{A}$)")
plt.savefig(FIG / "fig2_message_passing.png", dpi=170, bbox_inches="tight"); plt.close()

# ---- Fig 3: architecture diagram
R = json.load(open(ROOT / "outputs" / "results.json"))   # small synthetic SBM graph (supplementary)
fig, ax = plt.subplots(figsize=(13, 4.2)); ax.axis("off"); ax.set_xlim(0, 20); ax.set_ylim(0, 6)
def box(x, w, h, txt, fc, ec, y0=1.8):
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=9.5)
def arr(x1, x2, y=3.0): ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=14, color="#444"))
box(0.2, 2.2, 2.4, "Input\n$X$\n$N\\times F$", "#cfe0ff", "#2a6fdb"); arr(2.5, 3.2)
box(3.2, 3.8, 2.4, "GCN Layer 1\n$H^{(1)}=\\mathrm{ReLU}(\\hat{A}XW^{(1)}+b^{(1)})$\n$W^{(1)}: F\\times h_1$", "#ffe3c9", "#e8743b"); arr(7.1, 7.8)
box(7.8, 3.8, 2.4, "GCN Layer 2\n$H^{(2)}=\\mathrm{ReLU}(\\hat{A}H^{(1)}W^{(2)}+b^{(2)})$\n$W^{(2)}: h_1\\times h_2$", "#ffe3c9", "#e8743b"); arr(11.7, 12.4)
box(12.4, 3.8, 2.4, "GCN Layer 3\n$Z=\\hat{A}H^{(2)}W^{(3)}+b^{(3)}$\n$W^{(3)}: h_2\\times C$ (no ReLU)", "#d8f1ea", "#2ca58d"); arr(16.3, 17.0)
box(17.0, 2.8, 2.4, "softmax → $P$\n$N\\times C$\nloss on labelled\nnodes only", "#eee", "#555")
ax.text(10, 5.5, "Fixed propagation matrix $\\hat{A}=\\tilde{D}^{-1/2}(A+I)\\tilde{D}^{-1/2}$ is shared by all three layers (computed once)",
        ha="center", fontsize=10.5)
ax.text(10, 0.9, "Dropout on each layer input (training only) · Adam optimiser · cross-entropy + L2 on $W^{(1)}$ · exactly 3 graph-convolution layers",
        ha="center", fontsize=9.5, color="#444")
plt.savefig(FIG / "fig3_architecture.png", dpi=170, bbox_inches="tight"); plt.close()

# ---- Fig 4/5: SBM graph true vs predicted
D = np.load(ROOT / "outputs" / "results.npz"); A2, y, P, tr, te = D["A"], D["y"], D["P"], D["tr"], D["te"]
G2 = nx.from_numpy_array(A2); pos2 = nx.spring_layout(G2, seed=3, k=0.45)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
pred = P.argmax(1)
for a, lab, t in ((ax[0], y, "Ground-truth communities"), (ax[1], pred, "GCN predictions (squares = labelled training nodes)")):
    nx.draw_networkx_edges(G2, pos2, ax=a, alpha=0.25, width=0.8)
    for c in range(3):
        idx = [i for i in range(len(y)) if lab[i] == c]
        nx.draw_networkx_nodes(G2, pos2, nodelist=idx, node_color=COL[c], node_size=170, ax=a,
                               edgecolors="white", label=f"class {c}")
    a.set_title(t, fontsize=10); a.axis("off")
trn = [i for i in range(len(y)) if tr[i]]
nx.draw_networkx_nodes(G2, pos2, nodelist=trn, node_shape="s", node_color="none", edgecolors="black", node_size=300, ax=ax[1], linewidths=1.6)
wrong = [i for i in range(len(y)) if pred[i] != y[i]]
if wrong: nx.draw_networkx_nodes(G2, pos2, nodelist=wrong, node_shape="x", node_color="black", node_size=120, ax=ax[1])
ax[0].legend(loc="lower left", frameon=False); plt.tight_layout()
plt.savefig(FIG / "fig4_community_graph_predictions.png", dpi=170); plt.close()

# ---- Fig 5: training curves
h = R["hist"]; ep = np.arange(1, len(h["train_loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(ep, h["train_loss"], label="train loss", color=COL[0]); ax[0].plot(ep, h["val_loss"], label="validation loss", color=COL[1])
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend(frameon=False); ax[0].set_title("Loss")
ax[1].plot(ep, h["train_acc"], label="train acc", color=COL[0]); ax[1].plot(ep, h["val_acc"], label="validation acc", color=COL[1])
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(frameon=False); ax[1].set_title("Accuracy")
plt.tight_layout(); plt.savefig(FIG / "fig5_training_curves.png", dpi=170); plt.close()

# ---- Fig 6: GCN vs MLP over 10 runs
g, m = np.array(R["gcn_runs"]), np.array(R["mlp_runs"])
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.bar([0, 1], [m.mean(), g.mean()], yerr=[m.std(), g.std()], color=[COL[1], COL[0]], capsize=5, width=0.55)
ax.scatter(np.zeros(10) + np.linspace(-.12, .12, 10), m, color="k", s=14, zorder=3)
ax.scatter(np.ones(10) + np.linspace(-.12, .12, 10), g, color="k", s=14, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(["3-layer MLP\n(A_hat = I, no graph)", "3-layer GCN"]); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.08)
ax.set_title("10 runs: different graphs, splits, initialisations")
plt.tight_layout(); plt.savefig(FIG / "fig6_gcn_vs_mlp.png", dpi=170); plt.close()
print("figures written to", FIG)

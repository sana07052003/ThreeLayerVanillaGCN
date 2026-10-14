"""Builds docs/GCN_Assignment_Report.pdf. All numbers are read from outputs/*.json|txt,
i.e. they are the numbers produced by running the code."""
import io, json, pathlib, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, Preformatted, KeepTogether)

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
_fd = "/usr/share/fonts/truetype/dejavu/"
for n, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
             ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(n, _fd + f))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"; FIG = OUT / "figures"
F = json.load(open(OUT / "fixed_demo.json")); CR = json.load(open(OUT / "cora_results.json"))
import sys; sys.path.insert(0, str(ROOT / "src"))
from cora_data import load_cora
F = {k: np.array(v) for k, v in F.items()}

ss = getSampleStyleSheet()
for _k in ("Heading1","Heading2"): ss[_k].fontName = "DV-B"
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontName="DV", fontSize=9.2, leading=13.4, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], keepWithNext=1, fontSize=17, textColor=colors.HexColor("#1f3a68"), spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], keepWithNext=1, fontSize=12.5, textColor=colors.HexColor("#2a6fdb"), spaceBefore=9, spaceAfter=4)
CAP = ParagraphStyle("c", parent=BODY, fontSize=8.5, textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=9)
CODE = ParagraphStyle("code", fontName="DVM", fontSize=7, leading=9, backColor=colors.HexColor("#f4f6f9"),
                      borderPadding=5, spaceAfter=8, leftIndent=4)
NOTE = ParagraphStyle("n", parent=BODY, backColor=colors.HexColor("#fff7e0"), borderPadding=6, borderColor=colors.HexColor("#e0b84c"),
                      borderWidth=0.6, spaceBefore=4, spaceAfter=10)
P = lambda t, s=BODY: Paragraph(t, s)

_eq_n = [0]
def eq(latex, num=None, size=13):
    """Render a LaTeX math string with matplotlib mathtext as an inline image."""
    fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, f"${latex}$", fontsize=size)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=260, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig); buf.seek(0)
    from PIL import Image as PI
    w, h = PI.open(buf).size; buf.seek(0)
    img = Image(buf, width=w / 260 * 72, height=h / 260 * 72)
    cells = [["", img, f"({num})" if num else ""]]
    t = Table(cells, colWidths=[1.0 * cm, 14 * cm, 1.8 * cm])
    t.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "CENTER"), ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t

def fig(name, width=16.5, cap=None):
    from PIL import Image as PI
    w, h = PI.open(FIG / name).size
    out = [Image(str(FIG / name), width=width * cm, height=width * cm * h / w)]
    if cap: out.append(P(cap, CAP))
    return KeepTogether(out)

def mat(M, name=None, fmt="{:.4f}", cw=1.55):
    rows = [[fmt.format(v) if (abs(v - round(v)) > 1e-12 or "{:.4f}" != fmt) else str(int(round(v))) for v in r] for r in M]
    t = Table(rows, colWidths=[cw * cm] * M.shape[1])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "DV", 8), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                           ("LINEBEFORE", (0, 0), (0, -1), 1, colors.black), ("LINEAFTER", (-1, 0), (-1, -1), 1, colors.black),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t

def side(items):
    """items = [(label, matrix), ...] laid out horizontally"""
    cells = [[P(f"<b>{l}</b>", ParagraphStyle('x', parent=BODY, fontSize=8.5, alignment=1)) for l, _ in items],
             [mat(M) for _, M in items]]
    t = Table(cells, hAlign="CENTER"); t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return t

def tbl(rows, widths, head=True, fs=8.8):
    rows = [[P(str(c), ParagraphStyle('t', parent=BODY, fontSize=fs, leading=fs + 3, spaceAfter=0)) for c in r] for r in rows]
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2d3")), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eefb")))
    t.setStyle(TableStyle(st)); return t

# ======================= data used in the text (all read from outputs/) =======================
A_, X_, y_, tr_, va_, te_ = load_cora(); C = 7
res = CR["res"]; seeds = CR["seeds"]; CMS = CR["cm"]; hist = CR["hist"]
def col(name, k): return np.array([r[k] for r in res[name]])
def ms(name, k): v = col(name, k); return v.mean(), v.std(ddof=1)
def pm(name, k, d=4): a, b = ms(name, k); return f"{a:.{d}f}"  , f"{b:.{d}f}"
cora_txt = open(OUT / "cora_output.txt").read().strip().splitlines()
CODE_S = ParagraphStyle("codes", parent=CODE, fontSize=5.9, leading=7.4)
n_gcn_tr = np.bincount(y_[tr_], minlength=C); n_gcn_va = np.bincount(y_[va_], minlength=C); n_gcn_te = np.bincount(y_[te_], minlength=C)
acc_g, acc_m = col("gcn", "acc"), col("mlp", "acc")
Mcm = {s: np.array(CMS[f"gcn_{s}"]) for s in seeds}

S = []
# ======================= COVER =======================
S += [Spacer(1, 2.2 * cm),
      P("Three-Layer Vanilla Graph Convolutional Network", ParagraphStyle("T", parent=H1, fontSize=25, leading=30, alignment=1)),
      P("Derivation, Implementation and Evaluation for Node Classification on the Cora Citation Network",
        ParagraphStyle("T2", parent=BODY, fontSize=13, leading=17, alignment=1, textColor=colors.HexColor("#555555"))),
      Spacer(1, 0.8 * cm), fig("fig1_toy_graph_matrices.png", 15.5), Spacer(1, 0.4 * cm),
      tbl([["Assignment phase", "Where in this report"],
           ["Phase 1 – Basic understanding of the Vanilla GCN", "Sections 1, 3, 4, 5 (graph basics, aggregation, ingredients, architecture, diagrams)"],
           ["Phase 2 – Mathematical concepts and algorithm notes", "Section 6 (derivation, three-layer equations, loss, back-propagation, pseudocode, worked example)"],
           ["Phase 3 – Implementation", "Sections 7, 8, 9 and the GitHub repository (NumPy + PyTorch code, outputs, results)"]], [6.4, 10.4], fs=9),
      Spacer(1, 0.5 * cm),
      P("<b>Integrity statement.</b> Every number in this report was produced by running the accompanying code and is stored in the "
        "<font face='DVM'>outputs/</font> folder of the repository. The trained results come from the NumPy implementation, which was executed. "
        "A PyTorch implementation of the same model is included, but it could <b>not</b> be run in the environment where this package was prepared, "
        "so no PyTorch result is claimed. Fixed-weight demonstrations (Section 6.9) are never presented as trained results.", NOTE), PageBreak()]

# ======================= 1 INTRODUCTION =======================
S += [P("1. Introduction and Problem Statement", H1),
      P("Many datasets are networks rather than tables or images: papers cite papers, users follow users, proteins interact with proteins. "
        "A node in such a network is described by its own attributes, but a second, equally informative source of evidence is <i>who it is connected to</i>. "
        "Ordinary classifiers see only the first source. A <b>Graph Convolutional Network (GCN)</b> combines both by repeatedly averaging each node's "
        "representation with those of its neighbours."),
      P("This project studies the vanilla GCN of Kipf and Welling with <b>exactly three graph-convolution layers</b>. The practical task is "
        "<b>semi-supervised node classification</b> on the Cora citation network: every node is a scientific paper described by a bag-of-words vector, "
        "an edge is a citation link, and the goal is to predict the paper's subject class when only 140 of the 2,708 papers are labelled."),
      P("Formally, let the graph be G = (V, E, X) with |V| = N nodes, adjacency matrix A ∈ {0,1}<super>N×N</super> and feature matrix X ∈ ℝ<super>N×F</super>. "
        "Labels y<sub>i</sub> ∈ {0,…,C−1} are known only for a small set V<sub>L</sub>. The model must learn f<sub>θ</sub>(A, X) → P ∈ [0,1]<super>N×C</super> so that "
        "argmax P<sub>i</sub> matches y<sub>i</sub> on nodes whose labels were never used during training."),
      P("<b>Scope of the experiment.</b> The report evaluates exactly the protocol that is implemented: the public Planetoid split of Cora (140 training, 500 validation, "
        "1,000 test nodes, supplied with the data and not re-drawn), ten random seeds for weight initialisation and dropout, and checkpoint selection by validation loss. "
        "A single dataset is used, so the results do not show how the model behaves on other graphs, and no hyper-parameter search was performed.", NOTE),
      P("1.1 Datasets used", H2),
      tbl([["Dataset", "Role", "Description"],
           ["Toy graph (4 nodes)", "Illustration and hand verification", "3 edges, 2 features. Used with <b>fixed, hand-chosen weights</b> to show every intermediate matrix. No learning, no accuracy claimed."],
           ["Cora (2,708 nodes)", "Trainable model and evaluation", "5,278 undirected edges, 1,433 binary word features, 7 classes, public 140/500/1000 split."],
           ["60-node block-model graph", "Supplementary (repository only)", "Synthetic 3-community graph used for the gradient check and a small visual demonstration; see README."]], [3.8, 4.4, 8.6]),
      P("2. Objectives", H1),
      P("1. Represent a citation network as a graph with node-feature matrix, adjacency matrix and labels.<br/>"
        "2. Derive the GCN propagation rule (self-loops, degree matrix, symmetric normalisation) and the equations of all three layers.<br/>"
        "3. Implement every mathematical operation explicitly in NumPy, one function per formula, and verify intermediate matrices on a toy graph.<br/>"
        "4. Derive back-propagation for the three layers by hand and verify it against numerical (finite-difference) gradients.<br/>"
        "5. Train the three-layer GCN on Cora with the public split, select checkpoints by validation loss, and report held-out results over ten seeds with mean and sample standard deviation.<br/>"
        "6. Quantify what the graph contributes by comparing against the identical network with message passing removed (Â = I).<br/>"
        "7. Report accuracy, macro precision/recall/F1, confusion matrices and learning curves, and state the limits of the evidence.<br/>"
        "8. Provide a PyTorch implementation of the same model and organise code, outputs and documentation in a repository.")]

# ======================= 3 DATASET =======================
S += [PageBreak(), P("3. Dataset and Graph Representation (Phase 1)", H1), P("3.1 Graph fundamentals", H2),
      P("A <b>graph</b> models entities and relations. <b>Nodes</b> are the entities, <b>edges</b> the relations. Unlike an image grid or a sentence, a graph has no canonical node ordering and "
        "nodes have different numbers of neighbours, so a convolution must be defined through the connectivity itself."),
      tbl([["Object", "Meaning", "In Cora"],
           ["Node set V", "the N entities", "2,708 papers"],
           ["Edge set E", "pairs of related nodes", "5,278 citation links, treated as undirected"],
           ["Adjacency matrix A", "A<sub>ij</sub> = 1 if i and j are connected; symmetric, zero diagonal", "2,708 × 2,708, stored sparsely (avg. degree 3.90)"],
           ["Feature matrix X", "row i = feature vector x<sub>i</sub> of node i", "2,708 × 1,433 word indicators, row-normalised"],
           ["Degree d<sub>i</sub>", "number of neighbours = row sum of A", "no isolated nodes in Cora"],
           ["Labels y", "class of every node, known only on V<sub>L</sub>", "7 classes, 140 labelled training nodes"]], [3.2, 7.2, 6.4]),
      P("3.2 What the Cora graph means and how it is prepared", H2),
      P("Each node is a machine-learning paper; its feature vector marks which of 1,433 dictionary words appear in it. An edge records that one paper cites another. "
        "Preparation steps (in <font face='DVM'>src/cora_data.py</font>): (i) the Planetoid files are read; (ii) edges are symmetrised so information can flow both ways along a citation "
        "and duplicates and self-references are removed (giving 5,278 unique edges); (iii) each feature row is divided by its sum (standard row normalisation); "
        "(iv) the fixed index sets for training, validation and test are taken from the files. Self-loops are <i>not</i> stored in A; they are added inside the normalisation (Section 6.1). "
        "The homophily of the graph – the fraction of edges joining two papers of the same class – is %.3f, which is the structural property that neighbourhood averaging exploits." %
        float(A_.multiply(y_[:, None] == y_[None, :]).sum() / A_.sum())),
      P("3.3 Fixed data partitions", H2),
      tbl([["Partition", "Nodes", "Share", "Per-class counts (class 0 … 6)", "Used for"],
           ["Training", int(tr_.sum()), f"{100*tr_.mean():.1f}%", str(n_gcn_tr.tolist()), "loss and gradients"],
           ["Validation", int(va_.sum()), f"{100*va_.mean():.1f}%", str(n_gcn_va.tolist()), "checkpoint selection"],
           ["Test", int(te_.sum()), f"{100*te_.mean():.1f}%", str(n_gcn_te.tolist()), "final reporting only"],
           ["Total", A_.shape[0], "100%", str(np.bincount(y_).tolist()), ""]], [2.4, 1.7, 1.7, 7.4, 3.6]),
      P("The partitions are the public split and are not re-drawn by the code. The training set is balanced (20 nodes per class); the test set is not, and the largest test class (class 3) has "
        "%d of the 1,000 test nodes (%.1f%%), which is the accuracy of a constant prediction of that class. Because the test set has 1,000 nodes, one additional correct prediction changes accuracy by 0.1 percentage point." %
        (n_gcn_te.max(), 100 * n_gcn_te.max() / 1000)),
      P("3.4 Interpretation of the class labels", H2),
      P("The data files provide the class as a one-hot column index 0–6. This report keeps the numeric names and does not attach subject names to them, because the mapping was not verified here."),
      ]

# ======================= 4 WHY GCN =======================
S += [P("4. Why a Graph Convolutional Network? (Phase 1)", H1),
      P("A fully connected network classifies each paper from its own words only. In a citation network, however, papers on one topic cite each other, so the neighbours of a paper "
        "carry evidence about its topic. A GCN uses this evidence through <b>neighbourhood aggregation</b>: in each layer a node replaces its representation by a weighted average of its "
        "own and its neighbours' representations, followed by a learnable transformation."),
      P("<b>Message passing in three steps.</b> (1) Every neighbour j, and the node itself, sends its current feature vector. (2) The receiver forms a weighted sum, the weight being 1/√(d̃<sub>i</sub>d̃<sub>j</sub>). "
        "(3) The sum is multiplied by the shared weight matrix W, a bias is added and a non-linearity is applied. After one layer information has travelled one hop; after three layers a node's output depends on all nodes within three hops."),
      fig("fig2_message_passing.png", 11.2, "Figure 1 – Message passing at node 1 of the toy graph. The red numbers are the actual entries of row 1 of Â."),
      tbl([["Ingredient", "Role", "Without it"],
           ["Self-loops (A + I)", "keep the node's own features in its update", "own information lost; only neighbours contribute"],
           ["Degree normalisation D̃<super>-1/2</super>(·)D̃<super>-1/2</super>", "stops high-degree nodes dominating; keeps scale stable across layers", "sums grow with degree; unstable activations"],
           ["Weight matrix W<super>(l)</super>", "learned linear map; changes width (1433 → 64 → 32 → 7)", "nothing to learn; only fixed smoothing"],
           ["ReLU", "non-linearity between layers", "three layers collapse to the single linear map Â³XW"],
           ["Bias b<super>(l)</super>", "per-feature offset", "slightly less flexible"]], [4.2, 6.6, 6.0]),
      P("<b>What this experiment does and does not show.</b> The common claim that using the graph helps is tested here directly: the same three-layer network is trained with Â replaced by the identity "
        "matrix (so no information crosses edges), under the same split, seeds and hyper-parameters (Section 8.5). No claim is made about other graph models or about other datasets.")]

# ======================= 5 ARCHITECTURE =======================
S += [P("5. Methodology and Overall Architecture (Phase 1)", H1),
      P("Workflow: load Cora → symmetrise, row-normalise features → compute Â once → three graph-convolution layers → softmax over 7 classes → cross-entropy on the 140 training nodes → "
        "pick the checkpoint with the lowest validation loss → report test metrics."),
      fig("fig3_architecture.png", 16.5, "Figure 2 – Data flow of the three-layer GCN. h<sub>1</sub> = 64 and h<sub>2</sub> = 32 in the Cora experiment."),
      P("Table 1 – Tensor dimensions (N = 2,708 nodes, C = 7 classes).", CAP),
      tbl([["Stage", "Input", "Output", "Operation"],
           ["GCN 1", "N × 1433", "N × 64", "dropout on input, Â aggregation, linear, ReLU"],
           ["GCN 2", "N × 64", "N × 32", "dropout on input, Â aggregation, linear, ReLU"],
           ["GCN 3", "N × 32", "N × 7", "dropout on input, Â aggregation, linear (logits, no ReLU)"],
           ["Softmax", "N × 7", "N × 7", "row-wise class probabilities; loss only on the 140 labelled nodes"]], [2.4, 3.0, 3.0, 8.4]),
      P("Table 2 – Trainable parameters, including biases.", CAP),
      tbl([["Layer", "Calculation", "Parameters"],
           ["GCN 1", "1433 × 64 + 64", f"{1433*64+64:,}"], ["GCN 2", "64 × 32 + 32", f"{64*32+32:,}"], ["GCN 3", "32 × 7 + 7", f"{32*7+7:,}"],
           ["Total", "", f"{CR['n_params']:,}"]], [3.0, 8.0, 5.8]),
      P("The same pre-computed Â is used by all three layers. Layer 3 directly outputs class scores, so no extra dense layer is added and there are exactly three graph convolutions. "
        "A cost of depth is over-smoothing: repeated averaging makes neighbouring representations similar. The assignment fixes depth at three, so other depths were not studied.")]

# ======================= 6 MATH =======================
S += [PageBreak(), P("6. Mathematical Model (Phase 2)", H1), P("6.1 Adjacency, self-loops, degree matrix and symmetric normalisation", H2),
      P("The graph is represented by A ∈ {0,1}<super>N×N</super> (A<sub>ij</sub> = 1 if i and j are connected) and X ∈ ℝ<super>N×F</super>. Self-loops are added so that every node is its own neighbour:"),
      eq(r"\tilde{A}=A+I_N", 1), P("The degree matrix of the augmented graph is diagonal:"),
      eq(r"\tilde{D}_{ii}=\sum_{j}\tilde{A}_{ij}=1+d_i", 2),
      P("The symmetric normalisation divides each connection by the geometric mean of the two endpoint degrees:"),
      eq(r"\hat{A}=\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2},\qquad \hat{A}_{ij}=\frac{\tilde{A}_{ij}}{\sqrt{\tilde{d}_i\,\tilde{d}_j}}", 3),
      P("<b>Toy example</b> (edges 0–1, 1–2, 1–3): the degrees with self-loops are %s, so Â<sub>10</sub> = 1/√(4·2) = 0.3536 and Â<sub>11</sub> = 1/4 = 0.25." % ", ".join(str(int(v)) for v in np.diag(F["D"]))),
      side([("A", F["A"]), ("Ã = A + I", F["At"]), ("D̃", F["D"]), ("Â", F["Ah"])]),
      P("6.2 Derivation of the graph-convolution equation", H2),
      P("<b>Step 1 – spectral convolution.</b> With the normalised Laplacian L = I − D<super>-1/2</super>AD<super>-1/2</super> = UΛU<super>T</super>, filtering a node signal x with g<sub>θ</sub> is"),
      eq(r"g_\theta \star x = U\, g_\theta(\Lambda)\, U^{T} x", 4),
      P("This needs an eigen-decomposition and is not local. <b>Step 2 – Chebyshev polynomials</b> of Λ̃ = 2Λ/λ<sub>max</sub> − I approximate the filter by a K-hop local polynomial of L. "
        "<b>Step 3 – first order</b> (K = 1, λ<sub>max</sub> ≈ 2):"),
      eq(r"g_{\theta'}\star x\approx \theta'_0 x-\theta'_1 D^{-1/2}AD^{-1/2}x", 5),
      P("<b>Step 4 – tie the two parameters</b> (θ = θ'<sub>0</sub> = −θ'<sub>1</sub>):"),
      eq(r"g_\theta\star x\approx\theta\left(I+D^{-1/2}AD^{-1/2}\right)x", 6),
      P("<b>Step 5 – renormalisation trick.</b> The operator in (6) has eigenvalues in [0, 2]; stacking layers can make values explode or vanish. Replacing it by Â of (3) (eigenvalues in (−1, 1]) "
        "and generalising from one scalar filter to F input and F' output channels (θ becomes W ∈ ℝ<super>F×F'</super>) gives the <b>GCN layer</b>:"),
      eq(r"H^{(l+1)}=\sigma\!\left(\hat{A}\,H^{(l)}\,W^{(l)}+\mathbf{1}\,{b^{(l)}}^{T}\right),\qquad H^{(0)}=X", 7),
      P("Row i of (7) is the node-level (message-passing) form:"),
      eq(r"h_i^{(l+1)}=\sigma\!\left(\sum_{j\in\mathcal{N}(i)\cup\{i\}}\frac{1}{\sqrt{\tilde d_i\tilde d_j}}\;h_j^{(l)}W^{(l)}+b^{(l)}\right)", 8),
      P("6.3 Activation and dropout", H2),
      eq(r"\mathrm{ReLU}(z)=\max(0,z),\qquad \mathrm{ReLU}'(z)=1\ \mathrm{if}\ z>0,\ \ 0\ \mathrm{otherwise}", 9),
      P("ReLU is applied element-wise after aggregation and transformation in layers 1 and 2. Inverted dropout (rate p = 0.5) is applied to the input of each layer during training only:"),
      eq(r"\mathrm{Drop}_p(U)=\frac{M\odot U}{1-p},\qquad M_{ij}\sim\mathrm{Bernoulli}(1-p)", 10),
      P("The mask acts on feature entries, not on nodes or edges, and is the identity at evaluation time."),
      P("6.4 Equations of the three layers", H2),
      eq(r"H^{(1)}=\mathrm{ReLU}\!\left(\hat{A}\,X\,W^{(1)}+b^{(1)}\right),\quad W^{(1)}\in\mathbb{R}^{1433\times 64}", 11),
      eq(r"H^{(2)}=\mathrm{ReLU}\!\left(\hat{A}\,H^{(1)}\,W^{(2)}+b^{(2)}\right),\quad W^{(2)}\in\mathbb{R}^{64\times 32}", 12),
      eq(r"Z=\hat{A}\,H^{(2)}\,W^{(3)}+b^{(3)},\quad W^{(3)}\in\mathbb{R}^{32\times 7}", 13),
      P("In one line: Z = Â · ReLU(Â · ReLU(ÂXW<super>(1)</super>) W<super>(2)</super>) · W<super>(3)</super>. Without the two ReLUs this equals Â³XW<super>(1)</super>W<super>(2)</super>W<super>(3)</super>, a linear model."),
      P("6.5 Output layer, loss function and parameter learning", H2),
      eq(r"P_{ic}=\frac{\exp(Z_{ic})}{\sum_{c'}\exp(Z_{ic'})}", 14),
      P("Training is semi-supervised: the cross-entropy uses only the labelled nodes V<sub>L</sub>, but Â lets their gradient reach every node within three hops. "
        "Weight decay is applied to the first layer's weights."),
      eq(r"\mathcal{L}=-\frac{1}{|V_L|}\sum_{i\in V_L}\log P_{i,y_i}+\frac{\lambda}{2}\,\|W^{(1)}\|_F^{2}", 15),
      P("<b>Back-propagation.</b> Let M<sub>L</sub> be the 0/1 training-node mask and Y the one-hot labels. The gradient of softmax followed by cross-entropy is simple, and Â is symmetric, so Â<super>T</super> = Â:"),
      eq(r"\delta^{(3)}=\frac{\partial\mathcal{L}}{\partial Z}=\frac{M_L\odot(P-Y)}{|V_L|}", 16),
      eq(r"\frac{\partial\mathcal{L}}{\partial W^{(l)}}=\left(\hat{A}H^{(l-1)}\right)^{T}\delta^{(l)},\qquad \frac{\partial\mathcal{L}}{\partial b^{(l)}}=\mathbf{1}^{T}\delta^{(l)}", 17),
      eq(r"\delta^{(l-1)}=\left[\hat{A}\,\delta^{(l)}\,{W^{(l)}}^{T}\right]\odot\mathrm{mask}^{(l)}\odot\mathbb{1}\!\left[Z^{(l-1)}>0\right]", 18),
      P("where mask<super>(l)</super> is the dropout mask of layer l. The L2 term adds λW<super>(1)</super> to ∂L/∂W<super>(1)</super>. Parameters are updated with <b>Adam</b> (implemented from scratch): "
        "m ← β<sub>1</sub>m + (1−β<sub>1</sub>)g, v ← β<sub>2</sub>v + (1−β<sub>2</sub>)g², θ ← θ − η·m̂/(√v̂ + ε), with bias-corrected m̂, v̂. "
        "Equations (16)–(18) are checked against finite differences in Section 8.1."),
      P("6.6 Evaluation metrics", H2),
      P("Let C<sub>ab</sub> be the number of test nodes of actual class a predicted as b. Accuracy = Σ<sub>a</sub>C<sub>aa</sub> / Σ<sub>a,b</sub>C<sub>ab</sub>. For each class c, precision = TP<sub>c</sub>/(TP<sub>c</sub>+FP<sub>c</sub>), "
        "recall = TP<sub>c</sub>/(TP<sub>c</sub>+FN<sub>c</sub>), F1<sub>c</sub> = 2PR/(P+R); undefined ratios are set to 0. <b>Macro</b> averages give all 7 classes equal weight; macro F1 is the mean of the seven class F1 scores. "
        "Over R seeds the mean is m = (1/R)Σm<sub>r</sub> and the <i>sample</i> standard deviation is s = √(Σ(m<sub>r</sub> − m)² / (R − 1)) with R = 10. "
        "This spread measures variation due to initialisation and dropout on one fixed split, not uncertainty about the wider population of papers."),
      P("6.7 Algorithm and pseudocode", H2),
      Preformatted("""Algorithm: three-layer vanilla GCN, semi-supervised node classification
Input : A (N x N, sparse), X (N x F), labels y on training set V_L, widths h1, h2, epochs T
Output: parameters of the epoch with the lowest validation loss; class probabilities P

 1  A_tilde <- A + I                       # self-loops
 2  d_i     <- sum_j A_tilde[i, j]         # degrees
 3  A_hat   <- D^{-1/2} A_tilde D^{-1/2}   # symmetric normalisation, computed once
 4  Initialise W_l with Xavier-uniform, b_l <- 0                (l = 1, 2, 3)
 5  for epoch = 1 .. T:
 6      Hin_1 <- X
 7      for l = 1 .. 3:                                         # FORWARD
 8          U_l <- dropout(Hin_l)
 9          Z_l <- A_hat @ U_l @ W_l + b_l
10          Hin_{l+1} <- ReLU(Z_l)  if l < 3  else  Z_l
11      P <- softmax(Z_3)
12      L <- cross_entropy(P[V_L], y[V_L]) + (lambda/2) ||W_1||^2
13      delta <- (P - Y) * mask_L / |V_L|                       # BACKWARD
14      for l = 3, 2, 1:
15          gW_l <- (A_hat @ U_l)^T @ delta ;   gb_l <- column-sum(delta)
16          if l > 1: delta <- (A_hat @ delta @ W_l^T) * dropout_mask_l * [Z_{l-1} > 0]
17      Adam update of all W_l, b_l
18      evaluate (no dropout): validation loss; save parameters if it is the lowest so far
19  restore the saved parameters; return softmax(Z_3) for every node""", CODE),
      P("6.8 Complexity", H2),
      P("With sparse Â holding |E|' = 2|E| + N non-zero entries, one layer costs O(|E|'·F<sub>in</sub> + N·F<sub>in</sub>·F<sub>out</sub>) operations: the first term is the aggregation, the second the dense transform. "
        "Cora's Â has %d non-zeros instead of 2,708² ≈ 7.3 million entries, which is why sparse storage is used." % (2 * 5278 + 2708)),
      P("6.9 Numerical worked example (fixed weights – no training)", H2),
      P("The 4-node toy graph has features X below. The weights are <b>hand-chosen constants</b> with zero biases. This demonstrates only the mechanics of the forward pass; "
        "<b>the weights are not trained and the resulting probabilities carry no accuracy information</b>."),
      side([("X", F["X"]), ("W<sup>(1)</sup> 2×3", F["W1"]), ("W<sup>(2)</sup> 3×2", F["W2"]), ("W<sup>(3)</sup> 2×2", F["W3"])]), Spacer(1, 4),
      P("<b>Hand calculation, node 1, layer 1.</b> Row 1 of Â is [1/√8, 1/4, 1/√8, 1/√8] = [0.3536, 0.25, 0.3536, 0.3536]."),
      P("(ÂX)<sub>1</sub> = 0.3536·[1,0] + 0.25·[0,1] + 0.3536·[1,1] + 0.3536·[2,0] = [%.4f, %.4f], matching the code. Multiplying by W<super>(1)</super> gives [%s]; "
        "ReLU turns the negative entry to 0, so H<super>(1)</super><sub>1</sub> = [%s]." % (*F["AX"][1], ", ".join(f"{v:.4f}" for v in F["Z1"][1]), ", ".join(f"{v:.4f}" for v in F["H1"][1]))),
      side([("ÂX", F["AX"]), ("ÂXW<sup>(1)</sup>", F["Z1"]), ("H<sup>(1)</sup>", F["H1"])]), Spacer(1, 3),
      side([("ÂH<sup>(1)</sup>", F["AH1"]), ("ÂH<sup>(1)</sup>W<sup>(2)</sup>", F["Z2"]), ("H<sup>(2)</sup>", F["H2"])]), Spacer(1, 3),
      side([("ÂH<sup>(2)</sup>", F["AH2"]), ("Z", F["Z3"]), ("P = softmax(Z)", F["P"])]), Spacer(1, 4),
      P("Checks visible in the output: ReLU clips negative entries in layers 1 and 2; every row of P sums to 1; the hub node 1 receives the largest aggregated values because it has three neighbours.")]

# ======================= 7 PROTOCOL =======================
cfg = CR["cfg"]
S += [PageBreak(), P("7. Training Protocol and Reproducibility", H1),
      P("Table 3 – Settings used in the recorded experiment.", CAP),
      tbl([["Setting", "Value"],
           ["Dataset", "Cora, public Planetoid split: 140 train / 500 validation / 1,000 test"],
           ["Layer widths", "1433 → 64 → 32 → 7 (three graph convolutions)"],
           ["Initialisation", "Xavier/Glorot uniform for weights, zeros for biases"],
           ["Dropout", f"{cfg['dropout']} on the input of every layer (training only)"],
           ["Optimiser", f"Adam (own implementation), learning rate {cfg['lr']}, β = (0.9, 0.999), ε = 1e-8"],
           ["Weight decay", f"{cfg['wd']} on W<super>(1)</super> (added to the gradient)"],
           ["Epochs", f"{cfg['epochs']} per seed, full batch; training is never stopped early"],
           ["Checkpoint rule", "lowest validation cross-entropy (equation 19)"],
           ["Seeds", ", ".join(map(str, seeds))],
           ["Baseline", "identical network and settings with Â = I (no edges used)"],
           ["Hardware / libraries", "CPU; Python, NumPy, SciPy (no deep-learning framework needed for the reported results)"]], [4.0, 12.8]),
      P("The selected epoch for run r is"),
      eq(r"t_r^{*}=\arg\min_{t\in\{1,\dots,200\}}\mathcal{L}_{\mathrm{val}}(\theta_{r,t}),", 19),
      P("where L<sub>val</sub> is the cross-entropy over the 500 validation nodes with dropout off. The test set supplies no gradient and plays no role in this choice. "
        "The hyper-parameters (widths, learning rate, dropout, weight decay) were fixed before the final runs and were not tuned on the test set; they were not searched systematically either, "
        "so a different setting could give different numbers. Seeds control weight initialisation and dropout masks; the data split is the same for every run."),
      P("Reproducibility: <font face='DVM'>python src/run_cora.py</font> regenerates every number in Section 8. Run time was roughly 40–70 s per seed on one CPU core.")]

# ======================= 8 RESULTS =======================
Mg, Mm = ms("gcn", "acc"), ms("mlp", "acc")
S += [PageBreak(), P("8. Experimental Results (Phase 3)", H1), P("8.1 Verification of the hand-derived gradients", H2),
      P("To confirm that equations (16)–(18) are implemented correctly, <font face='DVM'>tests/gradient_check.py</font> compares them with central finite differences (step 10<super>-5</super>) "
        "for 30 randomly selected entries across all six parameter tensors (dropout off, small graph). Output stored in <font face='DVM'>outputs/gradient_check_output.txt</font>:"),
      Preformatted("\n".join(open(OUT / "gradient_check_output.txt").read().strip().splitlines()[-2:]), CODE),
      P("8.2 Program output", H2),
      P("The listing is the unmodified output of <font face='DVM'>src/run_cora.py</font> (file <font face='DVM'>outputs/cora_output.txt</font>).")]
S += [Preformatted("\n".join(cora_txt), CODE_S)]

S += [P("8.3 Checkpoint selection", H2),
      P("Table 4 – Selected checkpoints of the GCN (lowest validation loss).", CAP),
      tbl([["Seed", "Selected epoch", "Best validation loss", "Validation accuracy at that epoch", "Train accuracy"]] +
          [[r["seed"], r["best_epoch"], f"{r['best_val_loss']:.4f}", f"{r['val_acc']:.4f}", f"{r['train_acc']:.4f}"] for r in res["gcn"]], [2.2, 3.0, 3.8, 4.8, 3.0]),
      P("The selected epochs range from %d to %d out of 200, so the rule is active: the last epoch is not automatically the best, and validation loss rises again later in training as the model fits the 140 training nodes (training accuracy is at or near 1.0)." %
        (col("gcn", "best_epoch").min(), col("gcn", "best_epoch").max())),
      P("8.4 Learning curves", H2),
      fig("cora_curves.png", 16.5, "Figure 3 – Training and validation loss/accuracy for seed 42 (full 200 epochs). The dotted line marks the selected checkpoint."),
      P("8.5 Individual test runs, mean and sample standard deviation", H2),
      P("Table 5 – Test metrics of the selected GCN checkpoints (1,000 test nodes). Macro metrics average over the 7 classes.", CAP),
      tbl([["Seed", "Test loss", "Accuracy", "Macro precision", "Macro recall", "Macro F1"]] +
          [[r["seed"], f"{r['test_loss']:.4f}", f"{r['acc']:.4f}", f"{r['macro_precision']:.4f}", f"{r['macro_recall']:.4f}", f"{r['macro_f1']:.4f}"] for r in res["gcn"]], [2.0, 2.8, 2.8, 3.4, 3.0, 2.8]),
      Spacer(1, 6), P("Table 6 – Mean ± sample standard deviation over 10 seeds, GCN versus the same network without the graph (Â = I).", CAP),
      tbl([["Metric", "No graph (Â = I)", "Three-layer GCN"]] +
          [[lab, "%s ± %s" % pm("mlp", k), "%s ± %s" % pm("gcn", k)] for lab, k in
           (("Best validation loss", "best_val_loss"), ("Test loss", "test_loss"), ("Accuracy", "acc"), ("Macro precision", "macro_precision"),
            ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1"))], [5.0, 5.9, 5.9]),
      fig("cora_gcn_vs_mlp.png", 12.5, "Figure 4 – Test accuracy per seed, with and without message passing."),
      P("Averaged over ten seeds the GCN reaches %.1f%% test accuracy (s = %.1f points) against %.1f%% (s = %.1f points) for the identical network without edges, a difference of %.1f points; "
        "the GCN is ahead for each of the ten seeds (smallest per-seed margin %.1f points). The comparison uses the same split, features, widths and training settings, so the gap is attributable to neighbourhood aggregation under these settings. "
        "It is a statement about this split of Cora only; the standard deviations describe variation across seeds, not across datasets or splits." %
        (100 * Mg[0], 100 * Mg[1], 100 * Mm[0], 100 * Mm[1], 100 * (Mg[0] - Mm[0]), 100 * (acc_g - acc_m).min())),
      P("8.6 Confusion matrices and class-specific behaviour", H2),
      fig("cora_confusion.png", 16.5, "Figure 5 – Test confusion matrices of the GCN for the first three seeds. Rows: actual class; columns: predicted class."),
      ]
f1 = np.array(res["gcn"][0]["per_class_f1"]); M0 = Mcm[seeds[0]]
rec0 = np.diag(M0) / M0.sum(1)
S += [P("For seed %d the per-class F1 scores are %s; per-class recall is %s. The best-recognised class is %d and the weakest is %d. The largest single confusion is %d actual → %d predicted (%d nodes). "
        "A single accuracy value therefore hides substantial differences between classes." %
        (seeds[0], np.round(f1, 3).tolist(), np.round(rec0, 3).tolist(), int(rec0.argmax()), int(rec0.argmin()),
         *(lambda O: (int(np.unravel_index(O.argmax(), O.shape)[0]), int(np.unravel_index(O.argmax(), O.shape)[1]), int(O.max())))(M0 - np.diag(np.diag(M0))))),
      P("8.7 What the network learns: embeddings", H2),
      fig("cora_tsne.png", 16.5, "Figure 6 – t-SNE of all 2,708 nodes, coloured by true class: raw input features (left) and the 32-dimensional layer-2 representation of the seed-42 GCN (right). t-SNE is a visual aid; distances in it are not quantitative."),
      P("The right panel shows more compact class groups than the left. This is an illustration, not a measurement: t-SNE depends on its settings and no cluster-quality score was computed.")]

# ======================= 9 CODE =======================
S += [PageBreak(), P("9. Model Code and Formula-to-Code Map", H1),
      P("The core of the trainable model (<font face='DVM'>src/gcn_numpy.py</font>, excerpt). Each marked line implements one equation."),
      Preformatted('''def forward(self, A_hat, X, train=False):
    H = X
    for l in range(3):
        if train: m = (rng.random(H.shape) > p) / (1 - p); H = H * m   # Eq. 10  dropout
        AH = A_hat @ H                         # neighbourhood aggregation    (Eq. 7)
        Z  = AH @ self.W[l] + self.b[l]        # linear transform + bias
        H  = relu(Z) if l < 2 else Z           # Eq. 9 (layers 1-2); logits in layer 3
    return softmax(H)                          # Eq. 14

def backward(self, cache, y, train_mask):
    dZ = (P - Y) * train_mask[:, None] / n_tr  # Eq. 16
    for l in (2, 1, 0):
        gW[l] = cache["AH"][l].T @ dZ          # Eq. 17  (A_hat H)^T delta
        gb[l] = dZ.sum(axis=0, keepdims=True)
        if l > 0:
            dH = A.T @ (dZ @ self.W[l].T) * dropout_mask   # Eq. 18  A_hat symmetric
            dZ = dH * (cache["Z"][l - 1] > 0)              # ReLU derivative
    gW[0] += wd * self.W[0]                    # gradient of the L2 term in Eq. 15''', CODE),
      tbl([["Formula", "Code", "Note"],
           ["Ã = A + I  (1)", "gcn_ops.add_self_loops / sparse identity", "adds the diagonal"],
           ["D̃<sub>ii</sub> = Σ<sub>j</sub> Ã<sub>ij</sub>  (2)", "gcn_ops.degree_matrix, row sums", ""],
           ["Â = D̃<super>-1/2</super>ÃD̃<super>-1/2</super>  (3)", "gcn_ops.sym_normalize", "sparse diag(d<super>-1/2</super>) multiplied on both sides"],
           ["H' = σ(ÂHW + b)  (7)", "gcn_ops.gcn_layer; GCN3.forward", "<font face='DVM'>A_hat @ H</font> aggregates; <font face='DVM'>@ W</font> transforms"],
           ["ReLU (9), softmax (14)", "gcn_ops.relu, softmax", "softmax subtracts row max for stability"],
           ["Loss (15)", "gcn_ops.cross_entropy, GCN3.loss", "masked mean over labelled nodes + L2"],
           ["Gradients (16)–(18)", "GCN3.backward", "checked by tests/gradient_check.py"],
           ["Adam", "GCN3.step", "from scratch with bias correction"],
           ["Checkpoint rule (19)", "train_gcn(select='val_loss')", "parameters of lowest validation loss restored"],
           ["Metrics (Sec. 6.6)", "gcn_numpy.confusion_and_macro", "undefined ratios set to 0"]], [4.6, 6.0, 6.2], fs=8.2),
      Spacer(1, 6), P("Fixed-weight demonstration versus trainable GCN", H2),
      tbl([["", "Fixed-weight demo (Section 6.9)", "Trainable GCN (Section 8)"],
           ["File", "src/demo_fixed_weights.py", "src/gcn_numpy.py, src/run_cora.py"],
           ["Graph", "4 nodes, 2 features", "Cora: 2,708 nodes, 1,433 features, 7 classes"],
           ["Weights", "hand-picked constants, never updated", "Xavier init, Adam updates"],
           ["Meaningful predictions?", "No – checks arithmetic only", "Yes – evaluated on 1,000 held-out nodes"]], [3.6, 6.3, 6.9]),
      P("<b>PyTorch version.</b> <font face='DVM'>src/gcn_torch.py</font> defines the same model with autograd (<font face='DVM'>GraphConv</font> layers computing <font face='DVM'>A_hat @ H @ W + b</font>); "
        "<font face='DVM'>tests/test_torch_matches_numpy.py</font> checks that its forward pass and all gradients agree with the NumPy code. Neither file has been executed by the author of this package "
        "(no PyTorch installation was possible in the preparation environment); please run the test before using the PyTorch file.", NOTE)]

# ======================= CONCLUSION =======================
S += [P("Conclusion", H1),
      P("A three-layer vanilla GCN was derived from the spectral graph filter, implemented with one function per equation, with hand-derived back-propagation verified against numerical gradients, "
        "and evaluated on the public Cora split. Over ten seeds it obtained %.1f%% ± %.1f test accuracy and %.3f ± %.3f macro F1, compared with %.1f%% ± %.1f and %.3f ± %.3f for the same network without edges, "
        "so neighbourhood aggregation made a clear difference on this split." %
        (100 * Mg[0], 100 * Mg[1], *ms("gcn", "macro_f1"), 100 * Mm[0], 100 * Mm[1], *ms("mlp", "macro_f1"))),
      P("<b>Limits of the evidence.</b> One dataset and one fixed split were used; the seeds vary only initialisation and dropout. Hyper-parameters were not systematically tuned. "
        "No comparison with other graph architectures or with deeper or shallower GCNs was made, so nothing is claimed about them. Cora is strongly homophilous, which favours neighbourhood averaging; graphs where neighbours often have different labels may behave differently. "
        "The PyTorch implementation is untested here. The most informative next steps would be repeating the experiment with random splits or further datasets, and adding a depth study (which the three-layer constraint of this assignment excluded)."),
      P("How to run", H2),
      Preformatted("""pip install -r requirements.txt
bash run_all.sh                          # everything below, in order
python src/demo_fixed_weights.py         # fixed-weight demo (Section 6.9)
python tests/gradient_check.py           # gradient verification (Section 8.1)
python src/run_cora.py                   # main experiment, ~12 min on one CPU core
python src/make_figures.py; python src/make_cora_figures.py; python src/build_report.py
python tests/test_torch_matches_numpy.py # needs PyTorch; not run by the package author""", CODE),
      P("References", H2),
      P("T. N. Kipf and M. Welling, <i>Semi-Supervised Classification with Graph Convolutional Networks</i>, ICLR 2017. &nbsp; "
        "Z. Yang, W. Cohen, R. Salakhutdinov, <i>Revisiting Semi-Supervised Learning with Graph Embeddings</i>, ICML 2016 (Planetoid split of Cora). &nbsp; "
        "M. Defferrard, X. Bresson, P. Vandergheynst, <i>Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering</i>, NeurIPS 2016. &nbsp; "
        "D. K. Hammond, P. Vandergheynst, R. Gribonval, <i>Wavelets on graphs via spectral graph theory</i>, 2011.")]

def footer(c, d):
    c.saveState(); c.setFont("DV", 8); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Three-Layer Vanilla GCN – Derivation, Implementation and Evaluation"); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()

(ROOT / "docs").mkdir(exist_ok=True)
SimpleDocTemplate(str(ROOT / "docs" / "GCN_Assignment_Report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.8 * cm, bottomMargin=2 * cm, title="Three-Layer Vanilla GCN").build(S, onFirstPage=footer, onLaterPages=footer)
print("report written")

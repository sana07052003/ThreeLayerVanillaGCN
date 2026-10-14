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

import textwrap
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame

T2 = json.load(open(OUT / "two_layer_example.json"))
fx = {k: np.array(v) for k, v in T2["fixed"].items()}
yk, Pk, trk = np.array(T2["y"]), np.array(T2["P"]), np.array(T2["tr"]); predk = Pk.argmax(1)
hk = T2["hist"]; cmk = np.array(T2["cm"]); wrongk = [i for i in range(34) if predk[i] != yk[i]]
peak = max(hk["val_acc"]); peak_ep = hk["val_acc"].index(peak) + 1
sa = np.array(T2["seeds_acc"])
CODE_S = ParagraphStyle("codes", parent=CODE, fontSize=5.8, leading=7.0, borderPadding=3, spaceAfter=5)
OUT_S = ParagraphStyle("outs", parent=CODE_S, backColor=colors.HexColor("#eef7ee"))
H3 = ParagraphStyle("h3", parent=BODY, fontName="DV-B", fontSize=10, textColor=colors.HexColor("#1f3a68"), spaceBefore=6, spaceAfter=3, keepWithNext=1)

def listing(path, numbered=True, width=118):
    lines = open(ROOT / path).read().rstrip("\n").splitlines(); out = []
    for i, l in enumerate(lines, 1):
        parts = textwrap.wrap(l, width, subsequent_indent="      ", replace_whitespace=False, drop_whitespace=False) or [""]
        for k, part in enumerate(parts):
            out.append((f"{i:3d}  " if k == 0 else "     ") + ("» " + part.strip() if k else part))
    return Preformatted("\n".join(out), CODE_S)

def textfile(path):
    lines = open(ROOT / path).read().rstrip("\n").splitlines(); out = []
    for l in lines: out += textwrap.wrap(l, 125, subsequent_indent="   ", replace_whitespace=False, drop_whitespace=False) or [""]
    return Preformatted("\n".join(out), OUT_S)

def code_block(path, title, purpose, out_path=None, out_title="Output", extra=None):
    S2 = [P(f"File: <font face='DVM'>{path}</font> – {title}", H2), P(purpose), listing(path)]
    if extra:
        for ep, et in extra:
            S2 += [P(f"Code: <font face='DVM'>{ep}</font>", H3), listing(ep)]
    if out_path:
        S2 += [P(f"{out_title} (<font face='DVM'>{out_path}</font>)", H3), textfile(out_path)]
    return S2

class Doc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, **kw)
        self.addPageTemplates([PageTemplate(id="p", frames=[Frame(2 * cm, 2 * cm, A4[0] - 4 * cm, A4[1] - 3.8 * cm, id="f")], onPage=footer)])
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            self.notify("TOCEntry", (0 if f.style.name == "h1" else 1, f.getPlainText(), self.page))

def footer(c, d):
    c.saveState(); c.setFont("DV", 8); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Two-Layer Vanilla GCN – Mathematics, Example, Codes and Output"); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()

toc = TableOfContents(); toc.dotsMinLevel = 0
toc.levelStyles = [ParagraphStyle("t0", fontName="DV-B", fontSize=9.5, leading=13, leftIndent=0, spaceBefore=4),
                   ParagraphStyle("t1", fontName="DV", fontSize=8.3, leading=11, leftIndent=14)]
PARTS = ParagraphStyle("part", parent=H1, fontSize=22, leading=27, alignment=1, textColor=colors.HexColor("#1f3a68"))

S = []
# ===================== COVER =====================
S += [Spacer(1, 2.2 * cm), P("Two-Layer Vanilla Graph Convolutional Network", ParagraphStyle("T", parent=H1, fontSize=25, leading=30, alignment=1)),
      P("Mathematics, Algorithm, Working Code, Example, Output and Result<br/>followed by <b>Codes and Output</b> for every program of the project",
        ParagraphStyle("T2", parent=BODY, fontSize=12.5, leading=17, alignment=1, textColor=colors.HexColor("#555555"))),
      Spacer(1, 0.7 * cm), fig("two_layer_architecture.png", 16.2), Spacer(1, 0.3 * cm),
      tbl([["Part I – Two-Layer Vanilla GCN", "Introduction, aim, mathematics, algorithm, working code, example graphs, output, result"],
           ["Part II – Codes and Output", "Every program of the project (two-layer, three-layer, Cora, tests, PyTorch, workflow) with the output it produced"]], [5.6, 11.2], head=False, fs=9),
      Spacer(1, 0.4 * cm),
      P("<b>Companion document.</b> The three-layer GCN is treated in full in <i>GCN_Assignment_Report.pdf</i>; this document adds the two-layer model and gathers all code and outputs in one place.", BODY),
      P("<b>Integrity statement.</b> Every output shown here is the unmodified output of the program listed next to it, produced in the environment where this package was prepared. "
        "The two PyTorch files were <b>not</b> executed there; their output section is therefore left for the log of your own GitHub Actions run (Section 12). "
        "Fixed-weight examples are never presented as trained results.", NOTE), PageBreak(),
      P("Contents", H1), toc, PageBreak()]

# ===================== PART I =====================
S += [P("Part I – Two-Layer Vanilla GCN", PARTS), Spacer(1, 4), P("1. Introduction", H1),
      P("A <b>graph</b> G = (V, E) consists of nodes (entities) and edges (relations). Each node i carries a feature vector x<sub>i</sub>; stacking them gives the feature matrix X ∈ ℝ<super>N×F</super>, "
        "and the connections are stored in the adjacency matrix A ∈ {0,1}<super>N×N</super>. Many real problems – citation networks, social networks, molecules – are of this form, but a graph has no fixed node ordering and nodes have different numbers of neighbours, "
        "so convolution must be defined through the connections."),
      P("A <b>Graph Convolutional Network (GCN)</b> builds a representation for each node by repeatedly averaging the representations of the node and its neighbours and then transforming the average with learned weights. "
        "The aim is usually <b>semi-supervised node classification</b>: only a few nodes have labels, but because information flows along edges, the unlabelled nodes can be classified as well."),
      P("<b>Types of vanilla GCN by depth.</b> The number of graph-convolution layers L fixes how far information travels: after L layers a node's output depends on all nodes within L hops."),
      tbl([["Type", "Equations", "Receptive field", "Comment"],
           ["1-layer", "Z = Â X W", "1 hop", "a single linear filter; no hidden layer and no non-linearity"],
           ["<b>2-layer (this Part)</b>", "H = ReLU(ÂXW<super>(1)</super>), Z = ÂHW<super>(2)</super>", "2 hops", "the standard model of Kipf &amp; Welling: one hidden layer"],
           ["3-layer (companion report)", "H<super>(1)</super>, H<super>(2)</super> with ReLU, Z = ÂH<super>(2)</super>W<super>(3)</super>", "3 hops", "wider context; more parameters; more risk of over-smoothing"]], [3.2, 6.0, 2.4, 5.2], fs=8.4),
      P("This Part develops the two-layer model completely: the aim, the mathematics, the algorithm, the working code, a worked example, the program output and the result. Part II then lists every program written for the project, each followed by its output."),
      P("2. Aim", H1),
      P("To derive, implement and test a <b>two-layer vanilla GCN</b> for node classification using symmetric-normalised adjacency with self-loops, ReLU activation, softmax probabilities, cross-entropy loss, hand-derived back-propagation and the Adam optimiser. Concretely:"),
      P("1. Derive Â = D̃<super>-1/2</super>(A + I)D̃<super>-1/2</super> and the two-layer propagation equations.<br/>"
        "2. Derive the gradients of all parameters by hand and verify them against finite differences.<br/>"
        "3. Implement the model in NumPy, with one function per formula and no deep-learning framework.<br/>"
        "4. Demonstrate the arithmetic on a 4-node graph with fixed weights, and train the model on Zachary's Karate Club with only two labelled nodes.<br/>"
        "5. Report outputs and results without claiming more than the experiments show.")]

# ---- Mathematics
S += [P("3. Mathematics", H1), P("3.1 Graph representation, self-loops and degree matrix", H2),
      P("For a graph with N nodes, A<sub>ij</sub> = 1 if nodes i and j are connected and 0 otherwise (symmetric, zero diagonal). Adding self-loops lets a node keep its own features:"),
      eq(r"\tilde{A}=A+I_N", 1), eq(r"\tilde{D}_{ii}=\sum_{j}\tilde{A}_{ij}=1+d_i", 2),
      P("3.2 Symmetric normalisation", H2),
      P("Dividing each connection by the geometric mean of its endpoint degrees keeps the scale of aggregated features stable and treats i←j and j←i alike:"),
      eq(r"\hat{A}=\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2},\qquad \hat{A}_{ij}=\frac{\tilde{A}_{ij}}{\sqrt{\tilde{d}_i\,\tilde{d}_j}}", 3),
      P("3.3 Derivation of the graph-convolution layer", H2),
      P("<b>(i)</b> Spectral filtering of a node signal x with normalised Laplacian L = I − D<super>-1/2</super>AD<super>-1/2</super> = UΛU<super>T</super>: g<sub>θ</sub> ⋆ x = U g<sub>θ</sub>(Λ) U<super>T</super>x. "
        "<b>(ii)</b> A first-order Chebyshev approximation with λ<sub>max</sub> ≈ 2 gives g ⋆ x ≈ θ'<sub>0</sub>x − θ'<sub>1</sub>D<super>-1/2</super>AD<super>-1/2</super>x. "
        "<b>(iii)</b> Setting θ = θ'<sub>0</sub> = −θ'<sub>1</sub> gives θ(I + D<super>-1/2</super>AD<super>-1/2</super>)x. "
        "<b>(iv)</b> This operator has eigenvalues in [0, 2], which can destabilise stacked layers, so I + D<super>-1/2</super>AD<super>-1/2</super> is replaced by Â (renormalisation trick). "
        "<b>(v)</b> Generalising the scalar θ to a weight matrix W ∈ ℝ<super>F×F'</super> gives the layer"),
      eq(r"H^{(l+1)}=\sigma\!\left(\hat{A}\,H^{(l)}\,W^{(l)}+\mathbf{1}\,{b^{(l)}}^{T}\right),\qquad H^{(0)}=X", 4),
      eq(r"h_i^{(l+1)}=\sigma\!\left(\sum_{j\in\mathcal{N}(i)\cup\{i\}}\frac{1}{\sqrt{\tilde d_i\tilde d_j}}\;h_j^{(l)}W^{(l)}+b^{(l)}\right)", 5),
      P("Equation (5) is the node-level reading of (4): a degree-weighted sum of the neighbours' messages, then a shared linear map, bias and activation."),
      P("3.4 The two-layer model", H2),
      eq(r"H^{(1)}=\mathrm{ReLU}\!\left(\hat{A}\,X\,W^{(1)}+b^{(1)}\right),\quad W^{(1)}\in\mathbb{R}^{F\times h}", 6),
      eq(r"Z=\hat{A}\,H^{(1)}\,W^{(2)}+b^{(2)},\quad W^{(2)}\in\mathbb{R}^{h\times C},\qquad P=\mathrm{softmax}(Z)", 7),
      P("Combined: Z = Â · ReLU(ÂXW<super>(1)</super>) · W<super>(2)</super> (biases omitted). The ReLU, ReLU(z) = max(0, z) with derivative 1 if z &gt; 0 and 0 otherwise, is what prevents the two layers from collapsing into the single linear map Â²XW<super>(1)</super>W<super>(2)</super>. "
        "The second layer has no ReLU because it outputs class scores. Dimensions: Â is N×N, X is N×F, H<super>(1)</super> is N×h, Z and P are N×C."),
      P("3.5 Output layer and loss", H2),
      eq(r"P_{ic}=\frac{\exp(Z_{ic})}{\sum_{c'}\exp(Z_{ic'})}", 8),
      eq(r"\mathcal{L}=-\frac{1}{|V_L|}\sum_{i\in V_L}\log P_{i,y_i}+\frac{\lambda}{2}\,\|W^{(1)}\|_F^{2}", 9),
      P("The cross-entropy is computed only over the labelled nodes V<sub>L</sub> (semi-supervised); Â carries their influence to every node within two hops. λ is the weight-decay coefficient."),
      P("3.6 Back-propagation, derived step by step", H2),
      P("<b>Softmax + cross-entropy.</b> For one node, L<sub>i</sub> = −log P<sub>i,y</sub> with P = softmax(Z). Differentiating, ∂L<sub>i</sub>/∂Z<sub>ic</sub> = P<sub>ic</sub> − 1[c = y<sub>i</sub>]. Averaging over V<sub>L</sub> and zeroing unlabelled rows with the 0/1 mask M<sub>L</sub>:"),
      eq(r"\delta^{(2)}=\frac{\partial\mathcal{L}}{\partial Z}=\frac{M_L\odot(P-Y)}{|V_L|}", 10),
      P("<b>Layer 2.</b> Since Z = (ÂH<super>(1)</super>)W<super>(2)</super> + 1b<super>(2)T</super>:"),
      eq(r"\frac{\partial\mathcal{L}}{\partial W^{(2)}}=\left(\hat{A}H^{(1)}\right)^{T}\delta^{(2)},\qquad \frac{\partial\mathcal{L}}{\partial b^{(2)}}=\mathbf{1}^{T}\delta^{(2)},\qquad \frac{\partial\mathcal{L}}{\partial H^{(1)}}=\hat{A}^{T}\delta^{(2)}{W^{(2)}}^{T}=\hat{A}\,\delta^{(2)}{W^{(2)}}^{T}", 11),
      P("(Â is symmetric, so Â<super>T</super> = Â.) <b>ReLU.</b> The derivative passes the gradient only where the pre-activation was positive:"),
      eq(r"\delta^{(1)}=\frac{\partial\mathcal{L}}{\partial H^{(1)}}\odot\mathbb{1}\!\left[Z^{(1)}>0\right]", 12),
      P("<b>Layer 1.</b> Likewise:"),
      eq(r"\frac{\partial\mathcal{L}}{\partial W^{(1)}}=\left(\hat{A}X\right)^{T}\delta^{(1)}+\lambda W^{(1)},\qquad \frac{\partial\mathcal{L}}{\partial b^{(1)}}=\mathbf{1}^{T}\delta^{(1)}", 13),
      P("With dropout, the input of each layer is multiplied by its mask during training and the gradient flowing back through that input is multiplied by the same mask."),
      P("3.7 Optimiser and initialisation", H2),
      P("<b>Adam</b> keeps running estimates of the gradient mean m and of its square v for each parameter θ and updates"),
      eq(r"m\leftarrow\beta_1 m+(1-\beta_1)g,\quad v\leftarrow\beta_2 v+(1-\beta_2)g^2,\quad \theta\leftarrow\theta-\eta\,\frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)}+\epsilon}", 14),
      P("with β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.999, ε = 10<super>-8</super>. <b>Xavier-uniform initialisation</b> draws each weight from U(−a, a) with a = √(6/(F<sub>in</sub> + F<sub>out</sub>)), keeping activation variance roughly constant across layers; biases start at 0."),
      P("3.8 Parameters and cost", H2),
      P("The parameter count is F·h + h + h·C + C. For the Karate Club example (F = 34 identity features, h = 4, C = 2) this is 34·4 + 4 + 4·2 + 2 = %d. With a sparse Â holding |E'| non-zeros, one layer costs O(|E'|·F<sub>in</sub> + N·F<sub>in</sub>·F<sub>out</sub>)." % (34 * 4 + 4 + 4 * 2 + 2))]

# ---- Algorithm
S += [P("4. Algorithm", H1),
      Preformatted("""Algorithm: training a two-layer vanilla GCN (semi-supervised node classification)
Input : A (N x N), X (N x F), labels y on labelled set V_L, hidden width h, epochs T, learning rate eta
Output: trained parameters {W1, b1, W2, b2}; class probabilities P for every node

 1  A_tilde <- A + I                           # self-loops
 2  d_i     <- sum_j A_tilde[i, j]             # degrees
 3  A_hat   <- D^{-1/2} A_tilde D^{-1/2}       # symmetric normalisation (once)
 4  Initialise W1, W2 with Xavier-uniform; b1, b2 <- 0
 5  for epoch = 1 .. T:
 6      U1 <- dropout(X)                        # FORWARD
 7      Z1 <- A_hat @ U1 @ W1 + b1 ;  H1 <- ReLU(Z1)
 8      U2 <- dropout(H1)
 9      Z2 <- A_hat @ U2 @ W2 + b2 ;  P  <- softmax(Z2)
10      L  <- cross_entropy(P[V_L], y[V_L]) + (lambda/2) ||W1||^2
11      delta2 <- (P - Y) * mask_L / |V_L|      # BACKWARD
12      gW2 <- (A_hat @ U2)^T @ delta2 ;  gb2 <- column-sum(delta2)
13      delta1 <- (A_hat @ delta2 @ W2^T) * mask_2 * [Z1 > 0]
14      gW1 <- (A_hat @ U1)^T @ delta1 + lambda * W1 ;  gb1 <- column-sum(delta1)
15      Adam update of W1, b1, W2, b2
16      evaluate without dropout; keep the checkpoint chosen by the selection rule
17  return best parameters and softmax(Z2) for every node""", CODE),
      P("The checkpoint rule is a parameter of the code: <font face='DVM'>val_loss</font> (lowest validation cross-entropy), <font face='DVM'>val_acc</font>, or <font face='DVM'>last</font> (final epoch), used when no validation set exists, as in the Karate example below.")]

# ---- Working code
S += [P("5. Working Code", H1),
      P("The model is in <font face='DVM'>src/gcn_two_layer.py</font>; it builds on the helper functions of <font face='DVM'>src/gcn_ops.py</font> (Part II, Section 10). The table links each formula to the line of code that implements it."),
      tbl([["Formula", "Code", "Explanation"],
           ["Ã = A + I, Â = D̃<super>-1/2</super>ÃD̃<super>-1/2</super> (1)–(3)", "gcn_ops.sym_normalize", "self-loops, degrees, symmetric scaling"],
           ["H = σ(ÂHW + b) (4)", "GCN2.forward : AH = A_hat @ H ; Z = AH @ W + b", "aggregate, then transform"],
           ["ReLU (6)", "gcn_ops.relu", "np.maximum(0, Z) after layer 1 only"],
           ["softmax (8), loss (9)", "gcn_ops.softmax, cross_entropy, GCN2.loss", "stable softmax; masked mean + L2"],
           ["δ<super>(2)</super> (10)", "GCN2.backward : dZ = (P - Y) * mask / n", "softmax + cross-entropy gradient"],
           ["∂L/∂W<super>(2)</super>, ∂L/∂H<super>(1)</super> (11)", "gW[1] = AH[1].T @ dZ ; dH = A.T @ (dZ @ W[1].T)", "uses the cached ÂH from the forward pass"],
           ["δ<super>(1)</super> (12), ∂L/∂W<super>(1)</super> (13)", "dZ1 = dH * (Z[0] > 0) ; gW[0] = AH[0].T @ dZ1 + wd * W[0]", "ReLU derivative as a boolean mask"],
           ["Adam (14)", "GCN2.step", "from scratch with bias correction"]], [5.2, 6.6, 5.0], fs=8.2),
      P("Complete listing of <font face='DVM'>src/gcn_two_layer.py</font>:", BODY), listing("src/gcn_two_layer.py"),
      P("<b>Verification.</b> <font face='DVM'>tests/gradient_check_two_layer.py</font> compares equations (10)–(13) with central finite differences on 20 random parameter entries (dropout off). Result:"),
      Preformatted("\n".join(open(OUT / "gradient_check_two_layer_output.txt").read().strip().splitlines()[-2:]), OUT_S)]

# ---- Example
S += [P("6. Example Graphs", H1), P("6.1 Example 1 – four-node graph with fixed weights (arithmetic check, no training)", H2),
      P("Edges 0–1, 1–2, 1–3; two features per node. The weights below are <b>hand-chosen constants</b>; nothing is learned, so the probabilities say nothing about accuracy. The example shows every intermediate matrix of the two-layer forward pass."),
      side([("X", fx["X"]), ("W<sup>(1)</sup> 2×3", fx["W1"]), ("W<sup>(2)</sup> 3×2", fx["W2"])]), Spacer(1, 4),
      side([("Â", fx["Ah"]), ("ÂX", fx["AX"]), ("ÂXW<sup>(1)</sup>", fx["Z1"]), ("H<sup>(1)</sup> = ReLU", fx["H1"])]), Spacer(1, 4),
      side([("ÂH<sup>(1)</sup>", fx["AH1"]), ("Z = ÂH<sup>(1)</sup>W<sup>(2)</sup>", fx["Z2"]), ("P = softmax(Z)", fx["P"])]), Spacer(1, 4),
      P("Hand check of node 1: row 1 of Â is [0.3536, 0.25, 0.3536, 0.3536], so (ÂX)<sub>1</sub> = 0.3536·[1,0] + 0.25·[0,1] + 0.3536·[1,1] + 0.3536·[2,0] = [%.4f, %.4f], as in the output. "
        "ReLU zeroes the negative entries of layer 1, and every row of P sums to 1." % tuple(fx["AX"][1])),
      fig("fig1_toy_graph_matrices.png", 15.5, "Figure 1 – The four-node graph, Ã = A + I and Â."),
      P("6.2 Example 2 – Zachary's Karate Club (trained)", H2),
      P("Zachary's Karate Club is a classic social network: 34 club members (nodes) and 78 friendships (edges). The club later split into two factions, one around the instructor (node 0, ‘Mr. Hi’) and one around the administrator (node 33, ‘Officer’); each faction has 17 members. "
        "The task is binary node classification with <b>only two labelled nodes</b> (0 and 33); the other 32 members are unlabelled."),
      tbl([["Setting", "Value"],
           ["Graph", "34 nodes, 78 undirected edges; faction labels from the dataset (0 = Mr. Hi, 1 = Officer)"],
           ["Features", "identity matrix (X = I, F = 34): the model sees structure only, no member attributes"],
           ["Labelled / unlabelled", "2 labelled (nodes 0 and 33) / 32 unlabelled, used only for evaluation"],
           ["Architecture", "34 → 4 → 2, two graph convolutions, %d parameters" % (34 * 4 + 4 + 4 * 2 + 2)],
           ["Training", "Adam, learning rate 0.01, 300 epochs, no dropout, no weight decay, seed 42"],
           ["Checkpoint", "final epoch – with two labels there is no validation set to select by"],
           ["Evaluation", "accuracy and confusion matrix on the 32 unlabelled nodes at the final epoch"]], [4.0, 12.8])]

# ---- Output
log = open(OUT / "two_layer_example_output.txt").read().splitlines()
iB = [i for i, l in enumerate(log) if l.startswith("PART B")][0]; iN = [i for i, l in enumerate(log) if l.startswith("Node |")][0]
S += [P("7. Output", H1),
      P("The listing is the program output of <font face='DVM'>src/run_two_layer_example.py</font> for the Karate Club, from the training log up to the confusion matrix. The complete output, including the per-node probability table and the fixed-weight matrices, is reproduced in Part II."),
      Preformatted("\n".join(log[iB + 2:iN - 1]), OUT_S),
      P("Probability table for the first nodes and the ten-seed summary (same file):"),
      Preformatted("\n".join(log[iN:iN + 12] + ["   ..."] + log[-3:]), OUT_S),
      fig("two_layer_curves.png", 15.5, "Figure 2 – Left: training loss on the two labelled nodes. Right: accuracy on the 32 unlabelled nodes, monitored every epoch but never used for training or checkpoint choice."),
      fig("two_layer_karate_graph.png", 16.2, "Figure 3 – True factions (left) and the two-layer GCN's final-epoch prediction (right). Black squares: the two labelled nodes. Red rings: misclassified nodes."),
      fig("two_layer_probs.png", 16.2, "Figure 4 – Predicted probability of the ‘Officer’ faction for each member; bar colour is the true faction.")]

# ---- Result
S += [P("8. Result", H1),
      P("The two-layer vanilla GCN was implemented in NumPy with hand-derived gradients (finite-difference check passed) and trained on the Karate Club using only two labelled members. Results of the recorded run (seed 42, final epoch):"),
      tbl([["Quantity", "Value"],
           ["Training loss (2 labelled nodes)", "%.4f at epoch 1 → %.4f at epoch 300" % (hk["train_loss"][0], hk["train_loss"][-1])],
           ["Accuracy on the 32 unlabelled nodes, final epoch", "%.4f (%d of 32 correct)" % (T2["acc"], int(round(T2["acc"] * 32)))],
           ["Confusion matrix (rows actual, columns predicted)", "[[%d, %d], [%d, %d]]" % tuple(cmk.ravel())],
           ["Misclassified nodes", ", ".join(map(str, wrongk)) + " – all with true faction Mr. Hi" if all(yk[i] == 0 for i in wrongk) else ", ".join(map(str, wrongk))],
           ["Monitored accuracy on unlabelled nodes, highest value", "%.4f first reached at epoch %d" % (peak, peak_ep)],
           ["Ten seeds (0–9), final-epoch accuracy", "mean %.4f, sample std %.4f; values %s" % (sa.mean(), sa.std(ddof=1), np.round(sa, 4).tolist())]], [6.6, 10.2], fs=8.8),
      Spacer(1, 4),
      P("<b>Interpretation.</b> With a single labelled node per faction and no node attributes, the network recovers the two factions for most members purely from the edge structure, because information from nodes 0 and 33 spreads through two rounds of aggregation. "
        "Members that sit between the factions are the hardest; here all six errors are Mr. Hi members predicted as Officer. "),
      P("<b>Caution.</b> (1) Monitored accuracy was higher early in training (up to %.3f at epoch %d) and decreased to %.3f by epoch 300 as the model kept fitting its two labels. Picking the best epoch from this curve would use the unlabelled nodes' labels, so it is not done; the reported value is the final epoch, "
        "and the ten-seed mean (%.3f) shows how much results vary between initialisations. (2) The graph is tiny and the evaluation set has 32 nodes, so one node changes accuracy by 3.1 percentage points. "
        "(3) This example is not comparable with the Cora results of the three-layer report: different graph, features, label rate and protocol. No statement about which depth is better is made here." %
        (peak, peak_ep, hk["val_acc"][-1], sa.mean()), NOTE)]

# ===================== PART II =====================
S += [PageBreak(), P("Part II – Codes and Output", PARTS), Spacer(1, 4), P("9. Overview of the Programs", H1),
      P("All programs are in the GitHub repository. They fall into five groups: shared building blocks, the two-layer model, the three-layer model, tests, and automation. "
        "Each program below is listed in full, with line numbers, followed by the output it produced. Long lines are wrapped and marked with ‘»’."),
      tbl([["#", "File", "Purpose", "Output shown"],
           ["10", "src/graph_data.py, gcn_ops.py", "toy and synthetic graphs; formulas: self-loops, degrees, Â, ReLU, softmax, layer, loss", "–"],
           ["11", "src/gcn_two_layer.py + tests/gradient_check_two_layer.py", "two-layer GCN, Adam, checkpoint rule; gradient test", "gradient check"],
           ["12", "src/run_two_layer_example.py", "fixed-weight example + trained Karate Club", "full output"],
           ["13", "src/gcn_numpy.py + tests/gradient_check.py", "three-layer GCN; gradient test", "gradient check"],
           ["14", "src/demo_fixed_weights.py", "three-layer fixed-weight demo", "full output"],
           ["15", "src/cora_data.py, run_cora.py", "Cora loader; main 10-seed experiment", "full output"],
           ["16", "src/run_training.py", "supplementary run, 60-node synthetic graph", "full output"],
           ["17", "src/gcn_torch.py + tests/test_torch_matches_numpy.py", "PyTorch version and equivalence test", "from your Actions run"],
           ["18", "run_all.sh, .github/workflows/run.yml", "automation and the GitHub Actions workflow", "–"],
           ["19", "src/make_*figures.py, build_report.py", "figure and PDF generation (not listed; see repository)", "–"]], [0.9, 6.0, 7.2, 2.7], fs=8)]
S += [P("10. Building Blocks: Graph Data and Formulas", H1)]
S += code_block("src/graph_data.py", "datasets", "Generates the four-node toy graph used for fixed-weight examples and the 60-node, 3-community block-model graph used for the gradient checks and the supplementary run. Both are deterministic given a seed.")
S += code_block("src/gcn_ops.py", "one function per formula", "Implements Ã = A + I, the degree matrix, Â = D̃<super>-1/2</super>ÃD̃<super>-1/2</super> (dense and sparse), ReLU, a numerically stable softmax, the generic layer σ(ÂHW + b) and the masked cross-entropy.")
S += [P("11. Two-Layer Model and Gradient Test", H1), P("The model <font face='DVM'>src/gcn_two_layer.py</font> is listed in full in Section 5 and is not repeated here. Its gradient test follows.")]
S += code_block("tests/gradient_check_two_layer.py", "finite-difference test", "Compares the analytic gradients of all four parameter tensors with central differences (step 10<super>-5</super>).", "outputs/gradient_check_two_layer_output.txt")
S += [P("12. Two-Layer Worked Examples", H1)]
S += code_block("src/run_two_layer_example.py", "fixed-weight example and Karate Club", "Part A prints every matrix of the two-layer forward pass on the toy graph with fixed weights. Part B trains the two-layer GCN on the Karate Club (2 labelled nodes), prints the log, accuracy, confusion matrix, the per-node probabilities and a ten-seed summary.", "outputs/two_layer_example_output.txt")
S += [P("13. Three-Layer Model and Gradient Test", H1)]
S += code_block("src/gcn_numpy.py", "trainable three-layer GCN", "The model of the companion report: three graph convolutions, hand-derived back-propagation, Adam, dropout, L2 on the first layer, three checkpoint rules, confusion matrix and macro metrics, and a no-graph baseline (Â = I).")
S += code_block("tests/gradient_check.py", "finite-difference test", "Same test for the three-layer model on 30 parameter entries.", "outputs/gradient_check_output.txt")
S += [P("14. Three-Layer Fixed-Weight Demonstration", H1)]
S += code_block("src/demo_fixed_weights.py", "three-layer arithmetic demo (no training)", "Prints A, Ã, D̃, Â, the fixed weights and every intermediate matrix of the three-layer forward pass, plus a manual check of node 1.", "outputs/fixed_demo_output.txt")
S += [P("15. Cora: Data Loader and Main Experiment", H1)]
S += code_block("src/cora_data.py", "Cora loader", "Reads the Planetoid files, symmetrises the graph, row-normalises the features and builds the fixed 140/500/1000 split.")
S += code_block("src/run_cora.py", "ten-seed experiment", "Trains the three-layer GCN and the no-graph baseline for 10 seeds, selects the checkpoint with the lowest validation loss, and reports accuracy, macro metrics and a confusion matrix.", "outputs/cora_output.txt")
S += [P("16. Supplementary Run on the Synthetic Graph", H1)]
S += code_block("src/run_training.py", "60-node three-community graph", "Trains the three-layer GCN and the no-graph baseline on a synthetic block-model graph for 10 independently generated graphs. The task is easy by construction (strong community structure).", "outputs/training_output.txt")
S += [P("17. PyTorch Version and Equivalence Test", H1)]
S += code_block("src/gcn_torch.py", "PyTorch / autograd version", "The same three-layer model with autograd and <font face='DVM'>torch.optim.Adam</font>. Not executed in the environment where this package was prepared.")
S += code_block("tests/test_torch_matches_numpy.py", "equivalence test", "Loads the NumPy weights into the PyTorch model and checks that probabilities and all gradients agree.")
S += [P("Output", H3),
      P("<b>Not generated in the preparation environment.</b> Run the workflow step ‘PyTorch == NumPy equivalence test’ and ‘PyTorch GCN on Cora’ and paste the log lines here. "
        "Expected form of the test output: <font face='DVM'>PyTorch and NumPy implementations agree (forward pass + all gradients).</font> Until you paste your own log, nothing is claimed about these two files.", NOTE)]
S += [P("18. Automation", H1)]
S += code_block("run_all.sh", "run everything locally", "Runs the demos, tests, experiments, figures and report in order.")
S += code_block(".github/workflows/run.yml", "GitHub Actions workflow", "Runs the same steps on GitHub's servers when started manually from the Actions tab; the PyTorch steps are allowed to fail without stopping the rest.")
S += [P("19. Programs Not Listed", H1),
      P("<font face='DVM'>src/make_figures.py</font>, <font face='DVM'>src/make_cora_figures.py</font>, <font face='DVM'>src/make_two_layer_figures.py</font> draw the figures from the saved results, and <font face='DVM'>src/build_report.py</font> assembles the PDF reports. "
        "They contain no model logic and are omitted here for length; they are in the repository.")]

(ROOT / "docs").mkdir(exist_ok=True)
Doc(str(ROOT / "docs" / "Two_Layer_GCN_and_Codes.pdf"), pagesize=A4, title="Two-Layer Vanilla GCN and Codes").multiBuild(S)
print("written")

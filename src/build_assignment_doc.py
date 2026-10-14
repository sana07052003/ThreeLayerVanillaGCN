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
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, NextPageTemplate

# ---------------- cover fields: EDIT THESE ----------------
COURSE_CODE = "26AI632"; COURSE_NAME = "ADVANCED ALGORITHM DESIGN"
STUDENT_NAME = "Sana"; STUDENT_ID = "CB.SC.P2AIE26043"
# -----------------------------------------------------------

T2 = json.load(open(OUT / "two_layer_example.json")); fx = {k: np.array(v) for k, v in T2["fixed"].items()}
yk, Pk, trk = np.array(T2["y"]), np.array(T2["P"]), np.array(T2["tr"]); predk = Pk.argmax(1)
hk = T2["hist"]; cmk = np.array(T2["cm"]); wrongk = [i for i in range(34) if predk[i] != yk[i]]
peak = max(hk["val_acc"]); peak_ep = hk["val_acc"].index(peak) + 1; sa = np.array(T2["seeds_acc"])
A_, X_, y_, tr_, va_, te_ = load_cora(); C = 7
res = CR["res"]; seeds = CR["seeds"]; CMS = CR["cm"]; cfg = CR["cfg"]
def col(n, k): return np.array([r[k] for r in res[n]])
def ms(n, k): v = col(n, k); return v.mean(), v.std(ddof=1)
def pm(n, k, d=4): a, b = ms(n, k); return f"{a:.{d}f}", f"{b:.{d}f}"
acc_g, acc_m = col("gcn", "acc"), col("mlp", "acc"); Mg, Mm = ms("gcn", "acc"), ms("mlp", "acc")
n_tr, n_va, n_te = (np.bincount(y_[m], minlength=C) for m in (tr_, va_, te_))
cora_txt = open(OUT / "cora_output.txt").read().strip().splitlines()
D = np.load(OUT / "cora_seed42.npz"); Pc = D["P"]
CODE_S = ParagraphStyle("codes", parent=CODE, fontSize=6.3, leading=7.7, borderPadding=3, spaceAfter=5)
OUT_S = ParagraphStyle("outs", parent=CODE_S, backColor=colors.HexColor("#eef7ee"))
OUT_XS = ParagraphStyle("outxs", parent=OUT_S, fontSize=5.9, leading=7.1)
H3 = ParagraphStyle("h3", parent=BODY, fontName="DV-B", fontSize=9.6, spaceBefore=5, spaceAfter=2, keepWithNext=1)
BIG = ParagraphStyle("big", parent=H1, fontSize=30, leading=36, alignment=1, textColor=colors.black)
MID = ParagraphStyle("mid", parent=BODY, fontSize=14, leading=19, alignment=1)
def box(lead, text): return P(f"<b>{lead}</b> {text}", NOTE)
def listing(path, width=125):
    out = []
    for i, l in enumerate(open(ROOT / path).read().rstrip("\n").splitlines(), 1):
        parts = textwrap.wrap(l, width, subsequent_indent="      ", replace_whitespace=False, drop_whitespace=False) or [""]
        for k, part in enumerate(parts): out.append((f"{i:3d}  " if k == 0 else "     ") + ("» " + part.strip() if k else part))
    return Preformatted("\n".join(out), CODE_S)
def textfile(path, style=OUT_S):
    out = []
    for l in open(ROOT / path).read().rstrip("\n").splitlines(): out += textwrap.wrap(l, 132, subsequent_indent="   ", replace_whitespace=False, drop_whitespace=False) or [""]
    return Preformatted("\n".join(out), style)

class Doc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, **kw); fr = Frame(2 * cm, 1.9 * cm, A4[0] - 4 * cm, A4[1] - 4.1 * cm, id="f")
        mk = lambda name, head: PageTemplate(id=name, frames=[fr], onPage=lambda c, d, h=head: deco(c, d, h))
        self.addPageTemplates([mk("blank", None), mk("p2", "2-Layer Vanilla GCN Algorithm Laboratory"), mk("p3", "3-Layer Vanilla GCN Algorithm Laboratory")])
def deco(c, d, head):
    if head is None: return
    c.saveState(); c.setFont("DV", 8); c.setFillColor(colors.HexColor("#555555"))
    c.drawString(2 * cm, A4[1] - 1.5 * cm, head); c.setStrokeColor(colors.HexColor("#999999")); c.line(2 * cm, A4[1] - 1.65 * cm, A4[0] - 2 * cm, A4[1] - 1.65 * cm)
    c.drawCentredString(A4[0] / 2, 1.1 * cm, str(d.page)); c.restoreState()

S = []
# ============================== COVER ==============================
S += [Spacer(1, 2.5 * cm), P(COURSE_CODE, ParagraphStyle("c1", parent=MID, fontSize=16)), P(f"<b>{COURSE_NAME}</b>", ParagraphStyle("c2", parent=MID, fontSize=18)),
      Spacer(1, 1.2 * cm), P("<b>ASSIGNMENT</b>", ParagraphStyle("c3", parent=MID, fontSize=22, leading=28)), Spacer(1, 1.6 * cm),
      P("<b>TWO-LAYER VANILLA GCN</b>", ParagraphStyle("c4", parent=BIG, fontSize=26)), P("AND", MID), P("<b>THREE-LAYER VANILLA GCN</b>", ParagraphStyle("c5", parent=BIG, fontSize=26)),
      Spacer(1, 2.2 * cm), P("PREPARED BY", ParagraphStyle("c6", parent=MID, fontSize=12)), P(f"<b>{STUDENT_NAME}</b>", ParagraphStyle("c7", parent=MID, fontSize=16)), P(STUDENT_ID, MID),
      Spacer(1, 3.2 * cm),
      P("Every output in this document is the unmodified output of the program shown next to it. Fixed-weight examples are arithmetic demonstrations and are never presented as trained results. "
        "Code and outputs are in the GitHub repository ThreeLayerVanillaGCN.", ParagraphStyle("c8", parent=CAP, alignment=1)),
      NextPageTemplate("blank"), PageBreak()]
# ============================== PART 1 divider ==============================
S += [Spacer(1, 9 * cm), P("TWO-LAYER VANILLA GCN", BIG), NextPageTemplate("p2"), PageBreak()]

# ============================== TWO-LAYER ==============================
S += [P("Two-Layer Vanilla Graph Convolutional Network", ParagraphStyle("pt", parent=H1, fontSize=19, alignment=1, textColor=colors.black)),
      fig("two_layer_architecture.png", 16.0, "Figure 1: Architecture of the two-layer vanilla GCN."),
      P("1. Aim", H1),
      P("To implement a two-layer Vanilla Graph Convolutional Network (GCN) in Python (NumPy) for binary node classification using normalized adjacency with self-loops, ReLU activation, Softmax probabilities, "
        "cross-entropy loss, hand-derived backpropagation, and the Adam optimiser, and to test it on Zachary's Karate Club graph with only two labelled nodes."),
      P("2. Mathematics", H1), P("2.1 Graph, self-loops and normalized adjacency", H2),
      P("A graph with N nodes has adjacency matrix A (A<sub>ij</sub> = 1 if i and j are connected, otherwise 0) and node-feature matrix X ∈ ℝ<super>N×F</super>. Self-information is retained by adding the identity, and the degrees are taken from the augmented matrix:"),
      eq(r"\tilde{A}=A+I_N,\qquad \tilde{D}_{ii}=\sum_{j}\tilde{A}_{ij}", 1),
      eq(r"\hat{A}=\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2},\qquad \hat{A}_{ij}=\frac{\tilde{A}_{ij}}{\sqrt{\tilde{d}_i\,\tilde{d}_j}}", 2),
      P("This normalisation prevents high-degree nodes from dominating the aggregation and keeps the matrix symmetric."),
      P("2.2 Derivation of the graph-convolution layer", H2),
      P("Filtering a node signal x with the normalised Laplacian L = I − D<super>-1/2</super>AD<super>-1/2</super> = UΛU<super>T</super> gives g<sub>θ</sub> ⋆ x = U g<sub>θ</sub>(Λ) U<super>T</super>x. A first-order Chebyshev approximation with λ<sub>max</sub> ≈ 2 reduces this to "
        "θ'<sub>0</sub>x − θ'<sub>1</sub>D<super>-1/2</super>AD<super>-1/2</super>x; tying the parameters (θ = θ'<sub>0</sub> = −θ'<sub>1</sub>) gives θ(I + D<super>-1/2</super>AD<super>-1/2</super>)x. "
        "Because this operator has eigenvalues in [0, 2] and can destabilise stacked layers, it is replaced by Â of equation (2) (renormalisation trick), and the scalar θ is generalised to a weight matrix W:"),
      eq(r"H^{(l+1)}=\sigma\!\left(\hat{A}\,H^{(l)}\,W^{(l)}+\mathbf{1}\,{b^{(l)}}^{T}\right),\qquad h_i^{(l+1)}=\sigma\!\left(\sum_{j\in\mathcal{N}(i)\cup\{i\}}\frac{h_j^{(l)}W^{(l)}}{\sqrt{\tilde d_i\tilde d_j}}+b^{(l)}\right)", 3),
      P("The right-hand form shows the message-passing reading: each node takes a degree-weighted sum of its own and its neighbours' representations and applies the shared weights, a bias and an activation."),
      P("2.3 The two-layer model, softmax and loss", H2),
      eq(r"H^{(1)}=\mathrm{ReLU}\!\left(\hat{A}XW^{(1)}+b^{(1)}\right),\qquad Z=\hat{A}H^{(1)}W^{(2)}+b^{(2)},\qquad P=\mathrm{softmax}(Z)", 4),
      eq(r"\mathcal{L}=-\frac{1}{|V_L|}\sum_{i\in V_L}\log P_{i,y_i}+\frac{\lambda}{2}\,\|W^{(1)}\|_F^{2}", 5),
      P("ReLU(z) = max(0, z) is applied after layer 1 only; layer 2 outputs class scores. Without the ReLU the two layers would collapse into one linear map Â²XW<super>(1)</super>W<super>(2)</super>. Only labelled nodes V<sub>L</sub> enter the loss, "
        "but Â carries their influence to every node within two hops."),
      P("2.4 Back-propagation", H2),
      P("For softmax followed by cross-entropy, ∂L<sub>i</sub>/∂Z<sub>ic</sub> = P<sub>ic</sub> − 1[c = y<sub>i</sub>]. With the 0/1 labelled-node mask M<sub>L</sub> and one-hot labels Y, and using Â<super>T</super> = Â:"),
      eq(r"\delta^{(2)}=\frac{M_L\odot(P-Y)}{|V_L|},\qquad \frac{\partial\mathcal{L}}{\partial W^{(2)}}=\left(\hat{A}H^{(1)}\right)^{T}\delta^{(2)},\qquad \frac{\partial\mathcal{L}}{\partial b^{(2)}}=\mathbf{1}^{T}\delta^{(2)}", 6),
      eq(r"\delta^{(1)}=\left[\hat{A}\,\delta^{(2)}{W^{(2)}}^{T}\right]\odot\mathbb{1}\!\left[Z^{(1)}>0\right],\qquad \frac{\partial\mathcal{L}}{\partial W^{(1)}}=\left(\hat{A}X\right)^{T}\delta^{(1)}+\lambda W^{(1)},\qquad \frac{\partial\mathcal{L}}{\partial b^{(1)}}=\mathbf{1}^{T}\delta^{(1)}", 7),
      P("Adam updates every parameter θ from its gradient g with running moments m and v: m ← β<sub>1</sub>m + (1−β<sub>1</sub>)g, v ← β<sub>2</sub>v + (1−β<sub>2</sub>)g², θ ← θ − η·m̂/(√v̂ + ε), where m̂ and v̂ are the bias-corrected moments "
        "(β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.999, ε = 10<super>-8</super>). Weights start from Xavier-uniform U(−a, a), a = √(6/(F<sub>in</sub> + F<sub>out</sub>)). The parameter count is F·h + h + h·C + C."),
      box("Verification:", "equations (6)–(7) were checked against central finite differences on 20 randomly chosen parameter entries: " +
          open(OUT / "gradient_check_two_layer_output.txt").read().strip().splitlines()[-2].lower().replace("worst relative error over 20 checked entries", "worst relative error") + "; " +
          open(OUT / "gradient_check_two_layer_output.txt").read().strip().splitlines()[-1].lower() + "."),
      P("3. Algorithm", H1),
      Preformatted("""  1  procedure TwoLayerGCN(X, A, labels, labelled_mask, epochs, lr)
  2
  3      N <- number of nodes ; F <- input features ; h <- hidden units ; C <- classes
  4      Initialise W1 (F x h) and W2 (h x C) with Xavier-uniform, b1 <- 0, b2 <- 0, random seed fixed
  5
  6      A_tilde <- A + I                               # self-loops
  7      d_i     <- sum_j A_tilde[i, j]                 # degrees from A_tilde
  8      A_hat   <- D^(-1/2) * A_tilde * D^(-1/2)       # symmetric normalisation, once
  9
 10      for epoch <- 1 to epochs
 11          U1 <- dropout(X)                           # FORWARD
 12          Z1 <- A_hat * U1 * W1 + b1 ;  H1 <- ReLU(Z1)
 13          U2 <- dropout(H1)
 14          Z2 <- A_hat * U2 * W2 + b2 ;  P  <- Softmax(Z2)
 15          loss <- CrossEntropy(P[labelled], labels[labelled]) + (lambda/2) * ||W1||^2
 16
 17          delta2 <- (P - Y) * labelled_mask / |labelled|        # BACKWARD
 18          gW2 <- transpose(A_hat * U2) * delta2 ; gb2 <- column sums of delta2
 19          delta1 <- (A_hat * delta2 * transpose(W2)) * dropout_mask * [Z1 > 0]
 20          gW1 <- transpose(A_hat * U1) * delta1 + lambda * W1 ; gb1 <- column sums of delta1
 21
 22          Adam update of W1, b1, W2, b2
 23          evaluate without dropout ; keep the checkpoint chosen by the selection rule
 24
 25      P_final <- Softmax(A_hat * ReLU(A_hat * X * W1 + b1) * W2 + b2)
 26      for each node i
 27          predicted_class <- argmax(P_final[i]) ; display node, predicted class, probabilities
 28      display "Training completed"
 29      return predictions and probabilities""", CODE),
      P("4. Working Code", H1),
      P("The program consists of four Python files, listed in full. <font face='DVM'>graph_data.py</font> supplies the example graph, <font face='DVM'>gcn_ops.py</font> the formulas, <font face='DVM'>gcn_two_layer.py</font> the model, and "
        "<font face='DVM'>run_two_layer_example.py</font> the driver that prints the output of Section 6 (it also uses the helper <font face='DVM'>accuracy</font> from <font face='DVM'>gcn_numpy.py</font>, listed in the three-layer part)."),
      P("graph_data.py", H3), listing("src/graph_data.py"), P("gcn_ops.py", H3), listing("src/gcn_ops.py"),
      P("gcn_two_layer.py", H3), listing("src/gcn_two_layer.py"), P("run_two_layer_example.py", H3), listing("src/run_two_layer_example.py"),
      P("5. Example Graph", H1),
      P("<b>Example 1 – four-node graph with fixed weights.</b> The graph contains four nodes, two input features per node and edges E = {(0, 1), (1, 2), (1, 3)}. The weights are hand-chosen constants, so this example only checks the arithmetic of the forward pass."),
      tbl([["Node", "Feature 1", "Feature 2", "Neighbours"], ["0", "1.0", "0.0", "1"], ["1", "0.0", "1.0", "0, 2, 3"], ["2", "1.0", "1.0", "1"], ["3", "2.0", "0.0", "1"]], [2.0, 3.0, 3.0, 4.0], fs=9),
      Spacer(1, 4), P("The adjacency matrix is shown below with its normalised form. Self-loops are added before normalisation."),
      side([("A", np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]], float)), ("Â", fx["Ah"])]),
      fig("fig1_toy_graph_matrices.png", 15.0, "Figure 2: The four-node graph, Ã = A + I and Â."),
      P("<b>Example 2 – Zachary's Karate Club (trained).</b> The graph has 34 members (nodes) and 78 friendships (edges). The club later split into two factions of 17 members each, one around the instructor (node 0, ‘Mr. Hi’, class 0) and one around the administrator "
        "(node 33, ‘Officer’, class 1). Only these two nodes are labelled; the other 32 nodes are unlabelled and used for evaluation only. The input features are the identity matrix, so the model sees structure only."),
      tbl([["Setting", "Value"], ["Graph", "34 nodes, 78 edges, 2 classes (17 + 17)"], ["Features", "identity matrix, F = 34"], ["Labelled / unlabelled", "2 (nodes 0, 33) / 32"],
           ["Architecture", "34 → 4 → 2, %d parameters" % (34 * 4 + 4 + 4 * 2 + 2)], ["Training", "Adam, learning rate 0.01, 300 epochs, no dropout, no weight decay, seed 42"],
           ["Checkpoint", "final epoch (with two labels there is no validation set)"]], [4.2, 12.6], fs=9),
      P("6. Output", H1),
      P("The listing is the measured output of the program in Section 4 (<font face='DVM'>src/run_two_layer_example.py</font>), run with seed 42, 300 epochs and learning rate 0.01 for the trained example. Part A is the fixed-weight forward pass; Part B is the Karate Club."),
      textfile("outputs/two_layer_example_output.txt", OUT_XS),
      fig("two_layer_curves.png", 15.5, "Figure 3: Training loss on the two labelled nodes (left) and accuracy on the 32 unlabelled nodes (right). The unlabelled accuracy is monitored only; it never influences training or the choice of epoch."),
      fig("two_layer_karate_graph.png", 15.5, "Figure 4: True factions (left) and the final-epoch prediction (right). Black squares: labelled nodes. Red rings: misclassified nodes."),
      P("7. Result", H1),
      P("The two-layer Vanilla Graph Convolutional Network was implemented in Python (NumPy) for binary node classification. The model uses normalized adjacency with self-loops, two graph convolution layers, ReLU activation, Softmax probabilities, cross-entropy loss, "
        "backpropagation and Adam. Its gradients agree with finite differences."),
      P("The training loss on the two labelled nodes falls from %.4f at epoch 1 to %.4f at epoch 300. At the final epoch the network classifies %d of the 32 unlabelled members correctly (accuracy %.4f); the confusion matrix is [[%d, %d], [%d, %d]] (rows actual, columns predicted). "
        "The misclassified nodes are %s%s. Over ten seeds (0–9) the final-epoch accuracy is %.4f ± %.4f (mean ± sample standard deviation)." %
        (hk["train_loss"][0], hk["train_loss"][-1], int(round(T2["acc"] * 32)), T2["acc"], *cmk.ravel(), ", ".join(map(str, wrongk)),
         " (all true Mr. Hi members predicted as Officer)" if all(yk[i] == 0 for i in wrongk) else "", sa.mean(), sa.std(ddof=1))),
      box("Important interpretation:", "accuracy on the unlabelled nodes was higher early in training (%.3f first reached at epoch %d) and decreased to %.3f by epoch 300 as the model kept fitting its two labels. "
          "Choosing the best epoch from this curve would use the unlabelled labels, so the final epoch is reported. The graph is tiny (one node changes accuracy by 3.1 percentage points) and the example says nothing about larger graphs." % (peak, peak_ep, hk["val_acc"][-1]))]

# ============================== PART 2 divider ==============================
S += [NextPageTemplate("blank"), PageBreak(), Spacer(1, 9 * cm), P("THREE-LAYER VANILLA GCN", BIG), NextPageTemplate("p3"), PageBreak()]

# ============================== THREE-LAYER ==============================
S += [P("Three-Layer Vanilla Graph Convolutional Network", ParagraphStyle("pt3", parent=H1, fontSize=19, alignment=1, textColor=colors.black)),
      P("Subject Classification of Scientific Papers on the Cora Citation Network", ParagraphStyle("st3", parent=MID, fontSize=12.5)),
      P("1. Introduction and Problem Statement", H1),
      P("Scientific papers do not stand alone: each cites earlier work and is cited by later work, and papers on the same topic tend to cite one another. A paper's subject can therefore be estimated from its own words and also from the papers it is connected to. "
        "A model that looks only at the words ignores the second source of evidence."),
      P("This implementation models a collection of papers as a graph. Each paper is a node and a citation link is an edge. Every paper is described by a feature vector marking which of 1,433 dictionary words it contains. "
        "The practical problem is to decide a paper's subject class by combining its own words with the words of the papers it is connected to, when only a small fraction of the papers has known labels."),
      P("Let the citation graph be G = (V, E, X), where V is the set of papers (N = 2,708), E is the set of citation links (|E| = 5,278 undirected links after removing duplicates) and X ∈ ℝ<super>N×1433</super> is the feature matrix. "
        "The final model produces a probability vector p<sub>i</sub> over the seven subject classes for every paper."),
      box("Scope of the supplied experiment:", "the experiment uses the public Planetoid split of Cora (140 training, 500 validation and 1,000 test nodes, supplied with the data and not re-drawn), ten random seeds for weight initialisation and dropout, "
          "and checkpoint selection by validation loss. One dataset is used and no systematic hyper-parameter search was performed. The report therefore evaluates exactly the protocol that was run."),
      P("1.1. Dataset Used", H2),
      P("The experiment uses the Cora citation network in the Planetoid format. It has 2,708 papers, 5,278 citation links, 1,433 binary word features and 7 subject classes. Of the papers, 140 are labelled for training (20 per class), "
        "500 are used for validation and 1,000 for testing; the remaining nodes take part in message passing without their labels being used. The class convention is the numeric class index 0–6 provided with the data."),
      box("Dataset summary:", "the experiment operates on a 2,708-paper graph with 5,278 links. 140 papers supply the training loss, 500 select the checkpoint, and 1,000 are used for the final report. A small three-community synthetic graph and a four-node graph "
          "serve only for gradient checks and fixed-weight arithmetic demonstrations."),
      P("<b>Problem statement.</b> Given a citation graph G = (V, E, X) and labels on a few nodes, learn a function f<sub>θ</sub>(G, X) → (p<sub>1</sub>, p<sub>2</sub>, …, p<sub>N</sub>) such that the predicted class of every paper matches its subject. "
        "The central design requirement is that the prediction should not depend only on the paper's own words; it should also exploit the surrounding graph structure."),
      P("2. Objectives", H1),
      P("The main objective of the project is to implement and evaluate a three-layer Graph Convolutional Network for semi-supervised node classification. The concrete objectives are:"),
      P("1. Represent a citation network as a graph whose nodes are papers and whose edges are citation links.<br/>"
        "2. Encode each paper with a bag-of-words feature vector.<br/>"
        "3. Use three graph-convolution stages so that information can propagate from one-hop, two-hop and three-hop neighbourhoods.<br/>"
        "4. Produce a class-probability vector for every paper.<br/>"
        "5. Learn from only 140 labelled papers and classify the unlabelled papers (semi-supervised node classification).<br/>"
        "6. Derive the layer equations and the gradients by hand, and verify the gradients against finite differences.<br/>"
        "7. Select checkpoints by validation loss and report held-out results for ten seeds, with mean and sample standard deviation.<br/>"
        "8. Compare against the identical network without message passing to measure what the graph contributes.<br/>"
        "9. Report accuracy, macro precision, recall and F1, confusion matrices and the learning behaviour, and state the limits of the evidence."),
      P("3. Dataset and Graph Representation", H1), P("3.1. Meaning of the Graph", H2),
      P("In this experiment the collection of papers is represented as a graph in which each node is a paper. An edge records that one paper cites another; the links are treated as undirected so information can flow both ways. "
        "The graph contains 2,708 nodes and 5,278 links, with an average of 3.90 links per node and no isolated node. Each link is used by the graph convolution to let information move between neighbouring papers, so topology is part of the input, "
        "not only a collection of independent feature vectors."),
      P("The fraction of links joining two papers of the same class is %.3f. This homophily is the structural property that neighbourhood averaging exploits: a paper's neighbours are usually of its own class." %
        float(A_.multiply(y_[:, None] == y_[None, :]).sum() / A_.sum())),
      fig("fig2_message_passing.png", 10.5, "Figure 5: Message passing at one node: neighbour representations (and the node's own, through the self-loop) are weighted by the entries of the node's row of Â, summed, transformed and passed through ReLU."),
      P("3.2. Dataset and Preprocessing", H2),
      P("The principal structures are:<br/>• A: the 2,708 × 2,708 adjacency matrix, stored sparsely;<br/>• X: the 2,708 × 1,433 feature matrix, one row per paper;<br/>• y: the subject class of every paper, of which only the 140 training labels are visible to the loss;<br/>• P: the final class probabilities."),
      P("Preprocessing (<font face='DVM'>src/cora_data.py</font>): (i) the Planetoid files are read; (ii) edges are symmetrised, and duplicates and self-references are removed, giving 5,278 unique links; (iii) each feature row is divided by its sum (row normalisation); "
        "(iv) the fixed index sets for training, validation and test are taken from the files. Self-loops are not stored in A; they are added inside the normalisation (Section 6.1)."),
      P("3.3. Fixed Data Partitions", H2),
      P("The experiment uses the three conventional partitions supplied with the data."),
      tbl([["Partition", "Nodes", "Share", "Per-class counts (class 0 … 6)", "Purpose"],
           ["Training", int(tr_.sum()), f"{100*tr_.mean():.1f}%", str(n_tr.tolist()), "model supervision (loss, gradients)"],
           ["Validation", int(va_.sum()), f"{100*va_.mean():.1f}%", str(n_va.tolist()), "checkpoint selection"],
           ["Test", int(te_.sum()), f"{100*te_.mean():.1f}%", str(n_te.tolist()), "final evaluation only"],
           ["Total", A_.shape[0], "100%", str(np.bincount(y_).tolist()), "complete graph"]], [2.2, 1.6, 1.6, 7.2, 4.2], fs=8.4),
      P("The training set is balanced; the test set is not, and its largest class contains %d of the 1,000 test nodes (%.1f%%), which is the accuracy of a constant prediction of that class. "
        "With 1,000 test nodes, one additional correct prediction changes accuracy by 0.1 percentage point." % (n_te.max(), 100 * n_te.max() / 1000)),
      P("3.4. Interpretation of the Class Labels", H2),
      P("The data files provide each class as a one-hot column index from 0 to 6. This report keeps the numeric class names and does not attach subject names to them, because the mapping was not verified here. "
        "Accuracy and macro-averaged metrics can be computed correctly without semantic names; the final prediction of the model is the class with the largest probability, ŷ<sub>i</sub> = argmax<sub>c</sub> P<sub>ic</sub>."),
      P("4. Why a Graph Convolutional Network?", H1),
      P("A conventional fully connected classifier can process the word vector of each paper, but it does not represent the fact that papers cite one another. Subject matter is inherently relational: papers cite related work, and fields form clusters of mutually citing papers."),
      P("A GCN addresses this by repeatedly aggregating information from neighbouring nodes. The first layer uses direct one-hop neighbours, the second layer reaches two hops, and the third layer reaches three hops. "
        "The advantage can be illustrated by three papers A → B → C. After one graph-convolution layer, B can incorporate information from A. After two layers, C can incorporate information originating from A through B. A third layer extends the receptive field by another hop."),
      P("This is appropriate for paper classification because a paper whose words are ambiguous can be recognised by the papers around it. The architecture is also lightweight: the project uses a vanilla three-layer GCN rather than a deeper or more elaborate graph architecture, "
        "which keeps interpretation straightforward while allowing three-hop contextual reasoning."),
      tbl([["Ingredient", "Role", "Without it"],
           ["Self-loops (A + I)", "keep the node's own features in its update", "own information lost"],
           ["Degree normalisation", "stops high-degree nodes dominating; keeps scale stable across layers", "sums grow with degree; unstable activations"],
           ["Weight matrix W", "learned linear map; changes width 1433 → 64 → 32 → 7", "nothing to learn"],
           ["ReLU", "non-linearity between layers", "three layers collapse to the linear map Â³XW"]], [4.0, 7.2, 5.6], fs=8.4),
      box("Semi-supervised benefit:", "only 140 of the 2,708 papers are labelled. The graph convolution lets the label information of these papers influence their neighbours, so the model can classify papers without ever seeing their labels. "
          "Section 8.5 measures this benefit by training the identical network with the graph removed."),
      P("5. Methodology and Overall Architecture", H1),
      P("The overall workflow is:<br/>1. Construct the citation graph and the feature matrix.<br/>2. Row-normalise the features and symmetrise the graph.<br/>3. Normalise the adjacency matrix with self-loops.<br/>4. Apply GCN layer 1 to propagate one-hop information.<br/>"
        "5. Apply GCN layer 2 to propagate two-hop information.<br/>6. Apply GCN layer 3 to propagate three-hop information.<br/>7. Transform the final representation into class probabilities with softmax.<br/>8. Compute the loss on the 140 labelled nodes and update the parameters.<br/>"
        "9. Keep the checkpoint with the lowest validation loss and report the predicted class of every paper."),
      fig("fig3_architecture.png", 16.2, "Figure 6: Overall architecture. h<sub>1</sub> = 64 and h<sub>2</sub> = 32 in the Cora experiment."),
      P("5.1. Layer-by-Layer Implementation", H2),
      P("The model applies three graph-convolution stages. Layer 1 uses direct neighbours, Layer 2 uses two-hop information, and Layer 3 uses three-hop information. The conceptual computation is H<super>(0)</super> = X, "
        "H<super>(1)</super> = ReLU(ÂH<super>(0)</super>W<sub>1</sub>), H<super>(2)</super> = ReLU(ÂH<super>(1)</super>W<sub>2</sub>), H<super>(3)</super> = ÂH<super>(2)</super>W<sub>3</sub>, followed by P = softmax(H<super>(3)</super>)."),
      P("Table 1: Layer-wise interpretation and dimensions (N = 2,708 nodes, C = 7 classes).", CAP),
      tbl([["Stage", "Receptive field", "Shape", "Interpretation"],
           ["Raw features", "0-hop", "N × 1433", "A paper judged from its own words."],
           ["GCN Layer 1", "1-hop", "N × 64", "Direct citations influence the paper."],
           ["GCN Layer 2", "2-hop", "N × 32", "Two-hop context becomes visible."],
           ["GCN Layer 3", "3-hop", "N × 7", "The wider neighbourhood is combined; outputs class scores (no ReLU)."],
           ["Final output", "decision", "N × 7", "Softmax probability; class with the largest value is predicted."]], [3.0, 2.8, 2.6, 8.4], fs=8.4),
      Spacer(1, 3), P("Table 2: Trainable parameters, including biases.", CAP),
      tbl([["Layer", "Calculation", "Parameters"], ["GCN 1", "1433 × 64 + 64", f"{1433*64+64:,}"], ["GCN 2", "64 × 32 + 32", f"{64*32+32:,}"], ["GCN 3", "32 × 7 + 7", f"{32*7+7:,}"], ["Total", "", f"{CR['n_params']:,}"]], [3.0, 8.0, 5.8], fs=8.4),
      P("The trade-off at the third layer is that additional mixing can cause neighbouring papers to become more similar, which is the familiar over-smoothing issue in graph neural networks. The assignment fixes the depth at three, so other depths were not studied in this report."),
      P("6. Mathematical Model", H1), P("6.1. Adjacency Matrix and Self-Loops", H2),
      P("Let A ∈ ℝ<super>N×N</super> be the adjacency matrix of the citation graph. If papers i and j are connected, then A<sub>ij</sub> = 1; otherwise A<sub>ij</sub> = 0. Self-information is retained by adding the identity matrix; the degree matrix and the normalised adjacency matrix are"),
      eq(r"\tilde{A}=A+I,\qquad \tilde{D}_{ii}=\sum_{j}\tilde{A}_{ij},\qquad \hat{A}=\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}", 1),
      P("This normalisation prevents high-degree nodes from dominating the message aggregation. <b>Toy example</b> (edges 0–1, 1–2, 1–3): the degrees with self-loops are %s, so Â<sub>10</sub> = 1/√(4·2) = 0.3536 and Â<sub>11</sub> = 1/4 = 0.25." % ", ".join(str(int(v)) for v in np.diag(F["D"]))),
      side([("A", F["A"]), ("Ã = A + I", F["At"]), ("D̃", F["D"]), ("Â", F["Ah"])]),
      P("6.2. Graph Convolution", H2),
      P("For layer ℓ, let H<super>(ℓ)</super> denote the node representation. The derivation of the layer from the spectral graph filter is given in Part I, Section 2.2 (first-order Chebyshev approximation and the renormalisation trick). The result is"),
      eq(r"H^{(\ell+1)}=\mathrm{ReLU}\!\left(\hat{A}\,H^{(\ell)}\,W^{(\ell)}+\mathbf{1}\,{b^{(\ell)}}^{T}\right)", 2),
      eq(r"h_i^{(\ell+1)}=\mathrm{ReLU}\!\left(\sum_{j\in\mathcal{N}(i)\cup\{i\}}\hat{A}_{ij}\,h_j^{(\ell)}W^{(\ell)}+b^{(\ell)}\right),\qquad \hat{A}_{ij}=\frac{1}{\sqrt{\tilde d_i\tilde d_j}}", 3),
      P("Thus the same learned weight matrix is applied across the graph, while Â determines which neighbouring representations are aggregated. ReLU(z) = max(0, z) has derivative 1 for z &gt; 0 and 0 otherwise. "
        "During training, inverted dropout with rate 0.5, Drop<sub>p</sub>(U) = M ⊙ U/(1−p) with M<sub>ij</sub> ~ Bernoulli(1−p), is applied to the input of each layer; at evaluation time it is the identity."),
      P("6.3. Three-Layer Encoder", H2),
      eq(r"H^{(1)}=\mathrm{ReLU}(\hat{A}XW^{(1)}+b^{(1)}),\quad H^{(2)}=\mathrm{ReLU}(\hat{A}H^{(1)}W^{(2)}+b^{(2)}),\quad Z=\hat{A}H^{(2)}W^{(3)}+b^{(3)}", 4),
      P("with W<super>(1)</super> ∈ ℝ<super>1433×64</super>, W<super>(2)</super> ∈ ℝ<super>64×32</super>, W<super>(3)</super> ∈ ℝ<super>32×7</super>. The compact expression is Z = Â · ReLU(Â · ReLU(ÂXW<super>(1)</super>) W<super>(2)</super>) · W<super>(3)</super>. "
        "Without the two ReLUs it equals Â³XW<super>(1)</super>W<super>(2)</super>W<super>(3)</super>, a linear model, which is why the non-linearity matters."),
      P("6.4. Prediction Function", H2),
      eq(r"P_{ic}=\frac{\exp(Z_{ic})}{\sum_{c'}\exp(Z_{ic'})},\qquad \hat{y}_i=\arg\max_{c}P_{ic}", 5),
      P("For seven classes the final score is interpreted through a row-wise softmax; the predicted class is the one with the largest probability. The probabilities are model scores; their calibration is not tested."),
      P("6.5. Training Objective", H2),
      P("For labelled nodes i ∈ V<sub>L</sub> (here |V<sub>L</sub>| = 140), the cross-entropy objective with weight decay on the first layer is"),
      eq(r"\mathcal{L}=-\frac{1}{|V_L|}\sum_{i\in V_L}\log P_{i,y_i}+\frac{\lambda}{2}\,\|W^{(1)}\|_F^{2}", 6),
      P("Only the labelled nodes contribute to this objective, but Â lets their gradient reach every node within three hops. The gradients, derived by hand as in Part I and extended by one layer (M<sub>L</sub> the training mask, Y the one-hot labels), are"),
      eq(r"\delta^{(3)}=\frac{M_L\odot(P-Y)}{|V_L|},\qquad \frac{\partial\mathcal{L}}{\partial W^{(l)}}=\left(\hat{A}H^{(l-1)}\right)^{T}\delta^{(l)},\qquad \frac{\partial\mathcal{L}}{\partial b^{(l)}}=\mathbf{1}^{T}\delta^{(l)}", 7),
      eq(r"\delta^{(l-1)}=\left[\hat{A}\,\delta^{(l)}{W^{(l)}}^{T}\right]\odot\mathrm{mask}^{(l)}\odot\mathbb{1}\!\left[Z^{(l-1)}>0\right],\qquad l=3,2", 8),
      P("The parameters are updated with Adam (Part I, Section 2.4). The gradients (7)–(8) were verified against finite differences (Section 8.1)."),
      P("6.6. Evaluation Metrics", H2),
      P("Let C<sub>ab</sub> be the number of test nodes of actual class a predicted as b. Accuracy = Σ<sub>a</sub>C<sub>aa</sub>/Σ<sub>a,b</sub>C<sub>ab</sub>. For each class c let TP<sub>c</sub>, FP<sub>c</sub>, FN<sub>c</sub> be computed with c as the target; then"),
      eq(r"P_c=\frac{TP_c}{TP_c+FP_c},\quad R_c=\frac{TP_c}{TP_c+FN_c},\quad F1_c=\frac{2P_cR_c}{P_c+R_c},\quad F1_{\mathrm{macro}}=\frac{1}{7}\sum_{c=0}^{6}F1_c", 9),
      P("Macro precision and macro recall are the means of P<sub>c</sub> and R<sub>c</sub>; undefined ratios are set to zero. Over R = 10 seeds the mean is m = (1/R)Σm<sub>r</sub> and the sample standard deviation s = √(Σ(m<sub>r</sub> − m)²/(R − 1)). "
        "This spread measures variation due to initialisation and dropout on one fixed split, not uncertainty about the wider population of papers."),
      P("7. Training Protocol and Reproducibility", H1),
      P("The program output provides a complete training log and test metrics for every seed. Table 3 lists the settings that were used."),
      P("Table 3: Reproducibility information.", CAP),
      tbl([["Setting", "Value"], ["Model", "Vanilla three-layer GCN, widths 1433 → 64 → 32 → 7"], ["Network size", f"2,708 papers, 5,278 links, {CR['n_params']:,} trainable parameters"],
           ["Partitions", "public split: 140 training / 500 validation / 1,000 test"], ["Initialisation", "Xavier-uniform weights, zero biases"],
           ["Dropout", f"{cfg['dropout']} on the input of every layer (training only)"], ["Optimiser", f"Adam (own implementation), learning rate {cfg['lr']}"],
           ["Weight decay", f"{cfg['wd']} on W<super>(1)</super>"], ["Epochs", f"{cfg['epochs']} per seed, full batch, no early stopping"],
           ["Checkpoint criterion", "lowest validation cross-entropy"], ["Seeds", ", ".join(map(str, seeds))], ["Baseline", "same network with Â = I (no edges)"],
           ["Hardware / libraries", "CPU; Python, NumPy, SciPy (no deep-learning framework needed for the results)"]], [4.2, 12.6], fs=8.6),
      P("The selected epoch for run r is"),
      eq(r"t_r^{*}=\arg\min_{t\in\{1,\dots,200\}}\mathcal{L}_{\mathrm{val}}(\theta_{r,t}),", 10),
      P("where L<sub>val</sub> is the cross-entropy over the 500 validation nodes with dropout off. The test set supplies no gradient and plays no role in this choice."),
      box("Reproducibility limitation:", "hyper-parameters (widths, learning rate, dropout, weight decay) were fixed before the final runs and not tuned on the test set; they were not searched systematically either, so other settings could give different numbers. "
          "Seeds control weight initialisation and dropout masks; the data split is identical for every run. Running <font face='DVM'>python src/run_cora.py</font> regenerates every number of Section 8 (about 40–70 s per seed on one CPU core)."),
      P("8. Experimental Results", H1), P("8.1. Program Output", H2),
      P("First, the gradient check (<font face='DVM'>tests/gradient_check.py</font>) of the hand-derived back-propagation:"),
      Preformatted("\n".join(open(OUT / "gradient_check_output.txt").read().strip().splitlines()[-2:]), OUT_S),
      P("The main experiment (<font face='DVM'>src/run_cora.py</font>) output is reproduced below without modification."),
      Preformatted("\n".join(textwrap.fill(l, 135, subsequent_indent="   ") for l in cora_txt), OUT_XS)]

S += [P("8.2. Checkpoint Selection", H2),
      P("Table 4: Selected checkpoints of the GCN (lowest validation loss).", CAP),
      tbl([["Seed", "Selected epoch", "Best validation loss", "Validation accuracy", "Train accuracy"]] +
          [[r["seed"], r["best_epoch"], f"{r['best_val_loss']:.4f}", f"{r['val_acc']:.4f}", f"{r['train_acc']:.4f}"] for r in res["gcn"]], [2.2, 3.0, 3.8, 4.0, 3.0], fs=8.4),
      P("The selected epochs range from %d to %d out of 200, so the rule is active: the last epoch is not automatically the best, and validation loss rises again later as the model fits the 140 training nodes (training accuracy is at or near 1.0)." %
        (col("gcn", "best_epoch").min(), col("gcn", "best_epoch").max())),
      P("8.3. Learning Curves", H2),
      P("The training log records loss and accuracy on the training and validation nodes at every epoch. For seed 42 the training loss falls from %.3f to %.3f over 200 epochs, while the validation loss falls to its minimum at epoch %d and then fluctuates around a slightly higher level. "
        "The training accuracy reaches 1.0 early and the validation accuracy plateaus below it; this gap between falling training loss and fluctuating validation loss motivates selecting a checkpoint rather than keeping the last epoch." %
        (CR["hist"]["train_loss"][0], CR["hist"]["train_loss"][-1], CR["hist"]["best_epoch"])),
      fig("cora_curves.png", 16.2, "Figure 7: Training and validation loss and accuracy for seed 42. The dotted line marks the selected checkpoint."),
      P("8.4. Individual Test Runs", H2),
      P("At the selected checkpoint of each seed, the GCN produces the following results on the 1,000 test nodes."),
      P("Table 5: Test metrics of the selected GCN checkpoints. Macro metrics average over the 7 classes.", CAP),
      tbl([["Seed", "Test loss", "Accuracy", "Macro precision", "Macro recall", "Macro F1"]] +
          [[r["seed"], f"{r['test_loss']:.4f}", f"{r['acc']:.4f}", f"{r['macro_precision']:.4f}", f"{r['macro_recall']:.4f}", f"{r['macro_f1']:.4f}"] for r in res["gcn"]], [2.0, 2.8, 2.8, 3.4, 3.0, 2.8], fs=8.4),
      P("Accuracy ranges from %.3f to %.3f across seeds (%d to %d of 1,000 test nodes correct). Different seeds give different confusion matrices and different test losses, which is why the spread over seeds is reported next." %
        (acc_g.min(), acc_g.max(), int(round(acc_g.min() * 1000)), int(round(acc_g.max() * 1000)))),
      P("8.5. Mean and Sample Standard Deviation", H2),
      P("Table 6: Mean ± sample standard deviation over 10 seeds, GCN versus the same network without the graph (Â = I).", CAP),
      tbl([["Metric", "No graph (Â = I)", "Three-layer GCN"]] + [[lab, "%s ± %s" % pm("mlp", k), "%s ± %s" % pm("gcn", k)] for lab, k in
           (("Best validation loss", "best_val_loss"), ("Test loss", "test_loss"), ("Accuracy", "acc"), ("Macro precision", "macro_precision"), ("Macro recall", "macro_recall"), ("Macro F1", "macro_f1"))], [5.0, 5.9, 5.9], fs=8.8),
      fig("cora_gcn_vs_mlp.png", 11.5, "Figure 8: Test accuracy per seed, with and without message passing."),
      P("Averaged over ten seeds the GCN reaches %.1f%% test accuracy (s = %.1f points) against %.1f%% (s = %.1f points) for the identical network without edges, a difference of %.1f points; the GCN is ahead for every seed (smallest margin %.1f points). "
        "The comparison uses the same split, features, widths and settings, so the gap is attributable to neighbourhood aggregation under these settings. It is a statement about this split of Cora only." %
        (100 * Mg[0], 100 * Mg[1], 100 * Mm[0], 100 * Mm[1], 100 * (Mg[0] - Mm[0]), 100 * (acc_g - acc_m).min())),
      P("8.6. Confusion Matrix and Class-Specific Behaviour", H2),
      P("Rows represent actual classes and columns represent predicted classes. Figure 9 shows the matrices of the first three seeds.")]
Mc = np.array(CMS[f"gcn_{seeds[0]}"]); rec0 = np.diag(Mc) / Mc.sum(1); f10 = np.array(res["gcn"][0]["per_class_f1"]); Moff = Mc - np.diag(np.diag(Mc)); ia, ib = np.unravel_index(Moff.argmax(), Moff.shape)
S += [fig("cora_confusion.png", 16.2, "Figure 9: Test confusion matrices of the GCN for seeds 42, 123 and 456."),
      P("For seed %d, %d of the 1,000 test nodes are classified correctly. The per-class F1 scores are %s and the per-class recall values are %s. Class %d is recognised best and class %d worst; the largest single confusion is actual class %d predicted as class %d (%d nodes). "
        "A single accuracy value therefore hides differences between classes." %
        (seeds[0], int(np.trace(Mc)), np.round(f10, 3).tolist(), np.round(rec0, 3).tolist(), int(rec0.argmax()), int(rec0.argmin()), int(ia), int(ib), int(Moff.max()))),
      P("8.7. Full-Graph Demonstration", H2),
      P("Table 7 lists the first twenty test nodes (by index, not selected by outcome) for seed 42, with the true class, the predicted class and the probability of the predicted class."),
      P("Table 7: Predictions for the first twenty test nodes, seed 42.", CAP)]
tidx = np.where(te_)[0][:20]
S += [tbl([["Node", "True", "Predicted", "P(predicted)", "Result"]] + [[int(i), int(y_[i]), int(Pc[i].argmax()), f"{Pc[i].max():.3f}", "Correct" if Pc[i].argmax() == y_[i] else "Error"] for i in tidx], [2.4, 2.0, 2.6, 3.2, 2.4], fs=8.2),
      P("%d of these twenty are classified correctly. Over all 1,000 test nodes the seed-42 model is correct for %d. Figure 10 shows all 2,708 nodes in a two-dimensional t-SNE map, coloured by true class, for the raw input features and for the 32-dimensional layer-2 representation." %
        (int((Pc[tidx].argmax(1) == y_[tidx]).sum()), int((Pc[te_].argmax(1) == y_[te_]).sum()))),
      fig("cora_tsne.png", 16.2, "Figure 10: t-SNE of raw features (left) and of the layer-2 representation of the seed-42 GCN (right)."),
      box("Important interpretation:", "the probabilities are model scores, not proof of a paper's subject. The right-hand panel shows more compact class groups than the left, but t-SNE is a visual aid whose distances are not quantitative, and no cluster-quality score was computed. "
          "The results describe one dataset and one split; they should not be read as a general benchmark."),
      P("9. Model Code from the Implementation", H1),
      P("The final section reproduces the core model logic in mathematical and implementation-oriented form. The central model expression is Z = Â · ReLU(Â · ReLU(ÂXW<super>(1)</super>) W<super>(2)</super>) · W<super>(3)</super>, with Â = D̃<super>-1/2</super>(A + I)D̃<super>-1/2</super>. "
        "The following is the conceptual forward pass (a simplified form of the executed code):"),
      Preformatted('''# Three-layer GCN: conceptual forward pass
def gcn_layer(A_hat, H, W, b):
    return A_hat @ H @ W + b                      # aggregate, transform

H0 = X                                            # word features of all 2,708 papers
H1 = relu(gcn_layer(A_hat, H0, W1, b1))           # 1-hop information
H2 = relu(gcn_layer(A_hat, H1, W2, b2))           # 2-hop information
Z  = gcn_layer(A_hat, H2, W3, b3)                 # 3-hop information, class scores
P  = softmax(Z)                                   # class probabilities
prediction = P.argmax(axis=1)                     # predicted subject class''', CODE),
      P("The executed implementation is <font face='DVM'>src/gcn_numpy.py</font>, listed in full below, together with the formula-level helpers of <font face='DVM'>src/gcn_ops.py</font> (Part I) and the experiment driver <font face='DVM'>src/run_cora.py</font>."),
      P("gcn_numpy.py", H3), listing("src/gcn_numpy.py"), P("run_cora.py", H3), listing("src/run_cora.py"),
      P("Equivalent PyTorch model (<font face='DVM'>src/gcn_torch.py</font>, <b>not executed</b> in the environment where this document was prepared; an equivalence test <font face='DVM'>tests/test_torch_matches_numpy.py</font> is provided):"),
      Preformatted('''class GCN3(nn.Module):
    def forward(self, A_hat, X):
        H = F.relu(self.l1(A_hat, F.dropout(X, self.p, self.training)))   # A_hat @ H @ W + b
        H = F.relu(self.l2(A_hat, F.dropout(H, self.p, self.training)))
        return self.l3(A_hat, F.dropout(H, self.p, self.training))        # logits''', CODE),
      P("Conclusion", H1),
      P("A three-layer vanilla GCN was derived from the spectral graph filter, implemented with one function per equation and hand-derived back-propagation verified against numerical gradients, and evaluated on the public Cora split. "
        "The main strength of the approach is the use of topology together with word features: instead of judging a paper entirely in isolation, the GCN lets information propagate through the citation network, so that a paper can be classified from its neighbourhood, with only 140 labelled papers."),
      P("The experiment contains 2,708 papers, 5,278 links, 140 labelled training papers, 500 validation papers and 1,000 test papers. Over ten seeds the GCN obtained %.1f%% ± %.1f test accuracy and %.3f ± %.3f macro F1, compared with %.1f%% ± %.1f and %.3f ± %.3f for the same network without edges. "
        "The two-layer model of Part I was tested separately on the Karate Club graph and reached %.4f accuracy on its 32 unlabelled nodes at the final epoch." %
        (100 * Mg[0], 100 * Mg[1], *ms("gcn", "macro_f1"), 100 * Mm[0], 100 * Mm[1], *ms("mlp", "macro_f1"), T2["acc"])),
      P("The results are promising as a demonstration, but they should not be interpreted as a broad benchmark. One dataset and one fixed split were used, the seeds vary only initialisation and dropout, hyper-parameters were not systematically tuned, and no comparison with other graph architectures or other depths was made. "
        "The two examples of Part I and Part II use different graphs, features and label rates, so they say nothing about which depth is better. The strongest next experimental step would be to repeat the experiments over random splits and further datasets and to compare depths under one protocol."),
      P("Repository and how to run", H2),
      Preformatted("""pip install -r requirements.txt
python src/demo_fixed_weights.py        # three-layer fixed-weight demo
python tests/gradient_check.py          # three-layer gradient check
python tests/gradient_check_two_layer.py
python src/run_two_layer_example.py     # Part I examples
python src/run_cora.py                  # Part II experiment (~12 min on one CPU core)
python src/make_figures.py; python src/make_cora_figures.py; python src/make_two_layer_figures.py
python src/build_assignment_doc.py      # this document""", CODE)]

(ROOT / "docs").mkdir(exist_ok=True)
Doc(str(ROOT / "docs" / "GCN_Two_and_Three_Layer_Assignment.pdf"), pagesize=A4, title="Two-Layer and Three-Layer Vanilla GCN").build(S)
print("written")

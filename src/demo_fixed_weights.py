"""PART A - FIXED-WEIGHT numerical demonstration (NOT trained).
Weights are hand-chosen so that every number can be verified with a calculator.
Prints every intermediate matrix of the 3-layer forward pass."""
import json, pathlib, numpy as np
OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs"
from graph_data import toy_graph
from gcn_ops import add_self_loops, degree_matrix, sym_normalize, gcn_layer, relu, softmax

np.set_printoptions(precision=4, suppress=True, linewidth=120)
A, X, edges = toy_graph()
print("== Step 1: adjacency A and features X =="); print("A =\n", A); print("X =\n", X)
At = add_self_loops(A);            print("\n== Step 2: A_tilde = A + I ==\n", At)
Dt = degree_matrix(At);            print("\n== Step 3: degree matrix D_tilde ==\n", Dt)
Dinv = np.diag(1 / np.sqrt(np.diag(Dt))); print("\nD^-1/2 =\n", Dinv)
Ah = sym_normalize(A);             print("\n== Step 4: A_hat = D^-1/2 A_tilde D^-1/2 ==\n", Ah)
assert np.allclose(Ah, Dinv @ At @ Dinv) and np.allclose(Ah, Ah.T)

# fixed weights (2 -> 3 -> 2 -> 2), biases = 0
W1 = np.array([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]])
W2 = np.array([[1.0, -1.0], [1.0, 1.0], [-2.0, 1.0]])
W3 = np.array([[1.0, -1.0], [-1.0, 1.0]])
print("\nFixed weights\nW1 =\n", W1, "\nW2 =\n", W2, "\nW3 =\n", W3)

Z1, H1, AX = gcn_layer(Ah, X, W1)
print("\n== Layer 1 ==\nA_hat X =\n", AX, "\nA_hat X W1 (pre-ReLU) =\n", Z1, "\nH1 = ReLU(.) =\n", H1)
Z2, H2, AH1 = gcn_layer(Ah, H1, W2)
print("\n== Layer 2 ==\nA_hat H1 =\n", AH1, "\nA_hat H1 W2 (pre-ReLU) =\n", Z2, "\nH2 = ReLU(.) =\n", H2)
Z3, _, AH2 = gcn_layer(Ah, H2, W3, activation=None)
P = softmax(Z3)
print("\n== Layer 3 (output) ==\nA_hat H2 =\n", AH2, "\nLogits Z = A_hat H2 W3 =\n", Z3,
      "\nP = softmax(Z) =\n", P, "\nrow sums:", P.sum(1), "\npredicted class:", P.argmax(1))

# manual check of node 1, layer 1 (neighbours 0,1,2,3 incl. self-loop)
manual = sum(Ah[1, j] * X[j] for j in range(4))
assert np.allclose(manual, AX[1]); print("\nManual check node 1 aggregation OK:", manual)

json.dump({k: v.tolist() for k, v in dict(A=A, At=At, D=Dt, Ah=Ah, X=X, W1=W1, W2=W2, W3=W3,
           AX=AX, Z1=Z1, H1=H1, AH1=AH1, Z2=Z2, H2=H2, AH2=AH2, Z3=Z3, P=P).items()},
          open(str(OUT / "fixed_demo.json"), "w"), indent=1)

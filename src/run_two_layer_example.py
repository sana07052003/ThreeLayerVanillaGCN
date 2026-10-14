"""Two-layer GCN worked examples.
 Part A: fixed-weight forward pass on the 4-node toy graph (no training).
 Part B: TRAINED two-layer GCN on Zachary's Karate Club (34 members, 2 factions,
         only 2 labelled nodes: the instructor 0 and the administrator 33)."""
import json, pathlib, numpy as np, networkx as nx
from graph_data import toy_graph
from gcn_ops import sym_normalize, gcn_layer, softmax
from gcn_two_layer import train_two_layer
from gcn_numpy import accuracy

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs"
np.set_printoptions(precision=4, suppress=True, linewidth=120)

print("=" * 70, "\nPART A - fixed-weight two-layer forward pass (NOT trained)\n", "=" * 70, sep="")
A, X, _ = toy_graph(); Ah = sym_normalize(A)
W1 = np.array([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]); W2 = np.array([[1.0, -1.0], [1.0, 1.0], [-2.0, 1.0]])
print("A_hat =\n", Ah); Z1, H1, AX = gcn_layer(Ah, X, W1)
print("\nLayer 1: A_hat X =\n", AX, "\nA_hat X W1 =\n", Z1, "\nH1 = ReLU =\n", H1)
Z2, _, AH1 = gcn_layer(Ah, H1, W2, activation=None); P = softmax(Z2)
print("\nLayer 2: A_hat H1 =\n", AH1, "\nZ = A_hat H1 W2 =\n", Z2, "\nP = softmax(Z) =\n", P, "\nrow sums:", P.sum(1))
fixed = {k: v.tolist() for k, v in dict(Ah=Ah, X=X, W1=W1, W2=W2, AX=AX, Z1=Z1, H1=H1, AH1=AH1, Z2=Z2, P=P).items()}

print("\n" + "=" * 70, "\nPART B - trained two-layer GCN on Zachary's Karate Club\n", "=" * 70, sep="")
G = nx.karate_club_graph(); n = G.number_of_nodes()
A = nx.to_numpy_array(G, nodelist=range(n)); y = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)])
Xk = np.eye(n)                                    # identity features: the model sees structure only
tr = np.zeros(n, bool); tr[[0, 33]] = True; un = ~tr
print(f"nodes={n}  edges={G.number_of_edges()}  features={n} (identity)  classes=2")
print(f"faction sizes: Mr. Hi={int((y==0).sum())}, Officer={int((y==1).sum())}   labelled nodes: [0, 33]   unlabelled: {int(un.sum())}")
print("config: hidden=4, Adam lr=0.01, 300 epochs, no dropout, no weight decay, seed 42, checkpoint = final epoch\n" + "-" * 70)
model, fin, hist = train_two_layer(A, Xk, y, tr, un, hidden=4, epochs=300, lr=0.01, dropout=0.0, wd=0.0, seed=42,
                                   select="last", log_every=30)
# Only 2 labels exist, so there is no validation set: the FINAL epoch is used (select="last").
# The 32 unlabelled nodes are monitored for the learning curve but never influence training or checkpoint choice.
Pk = fin["P"]; acc_un = accuracy(Pk, y, un)
print("-" * 70, f"\nAccuracy on the 32 UNLABELLED nodes: {acc_un:.4f}  ({int((Pk.argmax(1)[un]==y[un]).sum())}/32)")
M = np.zeros((2, 2), int)
for a, p in zip(y[un], Pk.argmax(1)[un]): M[a, p] += 1
print("Confusion matrix (rows actual, cols predicted):\n", M)
print("\nNode | true | pred | P(Mr. Hi)  P(Officer) | labelled")
for i in range(n):
    print(f"{i:4d} | {y[i]:4d} | {Pk[i].argmax():4d} | {Pk[i,0]:9.4f}  {Pk[i,1]:9.4f} | {'yes' if tr[i] else ''}")
wrong = [i for i in range(n) if Pk[i].argmax() != y[i]]
print("\nMisclassified nodes:", wrong)

accs = []
for s in range(10):
    _, f, _ = train_two_layer(A, Xk, y, tr, un, hidden=4, epochs=300, lr=0.01, dropout=0.0, wd=0.0, seed=s, select="last")
    accs.append(accuracy(f["P"], y, un))
print(f"\n10 seeds, accuracy on unlabelled nodes: {np.round(accs,4).tolist()}  mean {np.mean(accs):.4f}  std {np.std(accs, ddof=1):.4f}")
json.dump(dict(fixed=fixed, P=Pk.tolist(), y=y.tolist(), tr=tr.tolist(), hist=hist, acc=acc_un, cm=M.tolist(),
               seeds_acc=accs, A=A.tolist(), H1=np.maximum(0, model.forward(sym_normalize(A), Xk)["Z"][0]).tolist()),
          open(OUT / "two_layer_example.json", "w"))

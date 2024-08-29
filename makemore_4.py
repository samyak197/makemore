import torch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Formatting the data
path = input("Enter file path :")
alphabets = set("abcdefghijklmnopqrstuvwxyz")
words = open(path, "r").read().splitlines()
words = [
    name
    for name in words
    if all(char in alphabets for char in name.lower()) and " " not in name
]
# words[0]="samyak"
words = list(set(words))
# Building the vocabulary of characters and maping to/from integers
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}


# Building the dataset
def build_dataset(words):
    block_size = 3

    X, Y = [], []

    for w in words:
        # print(w)
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context),'--->', itos[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


import random

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
# Dataset
Xtr.shape, Ytr.shape, Xdev.shape, Ydev.shape, Xte.shape, Yte.shape
# Hyperparameter for the model
block_size = 3
embedding_input = 2
emb_input = block_size * embedding_input
neurons = 6
outputs = 27
# ----

g = torch.Generator().manual_seed(19)
C = torch.randn((outputs, embedding_input), generator=g)
# Layer 1
W1 = torch.randn((emb_input, neurons), generator=g) * (5 / 3) / (emb_input**0.5)
b1 = torch.randn(neurons, generator=g) * 0.1
# Layer 2
W2 = torch.randn((neurons, outputs), generator=g) * 0.1
b2 = torch.randn(outputs, generator=g) * 0.1
# BatchNorm params
bngain = torch.randn((1, neurons)) * 0.1 + 1.0
bnbias = torch.randn((1, neurons)) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
total_elements = 0
for p in parameters:
    total_elements += p.nelement()
print(total_elements)
for p in parameters:
    p.requires_grad = True

# Hyperparameters
lri = []
stepi = []
lossi = []
batch_size = 16
max_steps = 50000
lrs_e = [0.1, 0.01]
ud = []
batch_size = 16
n = batch_size

with torch.no_grad():
    for i in range(max_steps):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        # Forward pass
        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        # Linear layer 1
        hprebn = embcat @ W1 + b1
        # BatchNorm layer
        bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
        bndiff = hprebn - bnmeani
        bndiff2 = bndiff**2
        bnvar = (
            1 / (n - 1) * (bndiff2).sum(0, keepdim=True)
        )  # Bessel's correction i.e dividing by (n-1) not n
        bnvar_inv = (bnvar + 1e-5) ** -0.5
        bnraw = bndiff * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        # Non Linearity
        h = torch.tanh(hpreact)  # hidden Layer

        # Linear Layer 2
        logits = h @ W2 + b2  # output layer

        loss = F.cross_entropy(logits, Yb)
        # Pytorch backward pass
        for p in parameters:
            p.grad = None

        # manual backward pass

        dlogits = F.softmax(logits, 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n
        # 2nd layer bp
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        # tanh
        dhpreact = (1.0 - h**2) * dh
        # batchnorm layer  bp
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = (
            bngain
            * bnvar_inv
            / n
            * (
                n * dhpreact
                - dhpreact.sum(0)
                - (n / (n - 1)) * bnraw * (dhpreact * bnraw).sum(0)
            )
        )
        # 1st layer
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = 1.0 * dhprebn.sum(0)
        # embedding
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for k in range(Xb.shape[0]):
            for j in range(Xb.shape[1]):
                ix = Xb[k, j]
                dC[ix] += demb[k, j]
        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

        # Update

        lr = 0.1 if i < 43000 else 0.01
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad

        # Track stats

        if i % 10000 == 0:
            print(f"{1:7d}/{max_steps:7d}:{loss.item():.4f}")
        lossi.append(loss.log10().item())
    print(loss.item())

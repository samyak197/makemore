import torch
import torch.nn.functional as F

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
embedding_input = 6
emb_input = 3 * embedding_input
neurons = 32
# ----

g = torch.Generator().manual_seed(19)
C = torch.randn((27, embedding_input), generator=g)
W1 = torch.randn((emb_input, neurons), generator=g)
b1 = torch.randn(neurons, generator=g)
W2 = torch.randn((neurons, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
param = 0
for p in parameters:
    param += p.nelement()

print(f"{param=}")

for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
lri = []
stepi = []
lossi = []
for i in range(25000):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # Forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, emb_input) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    # print(loss.item())

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update
    # lr=lrs[i]
    lr = 0.11
    for p in parameters:
        p.data += -lr * p.grad

    # Track stats
    # lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())

print(f"loss= {loss.item()}")
emb = C[Xdev]
h = torch.tanh(emb.view(-1, emb_input) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
emb = C[Xtr]
h = torch.tanh(emb.view(-1, emb_input) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
g = torch.Generator().manual_seed(19)

num = int(input("how many words do you want to generate ?"))
block_size = 3
for _ in range(num):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))

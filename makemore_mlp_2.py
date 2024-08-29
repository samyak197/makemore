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


class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g)  # / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNormId:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        # parameters
        self.eps = eps
        self.training = True
        self.momentum = momentum
        # buffers
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training is True:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            x_var = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        # Update the buffers
        if self.training is True:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * xmean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


vocab_size = 27
n_embed = 3
n_hidden = 6
g = torch.Generator().manual_seed(19)
block_size = 3

C = torch.randn((vocab_size, n_embed), generator=g)

layers = [
    Linear(n_embed * block_size, n_hidden),
    BatchNormId(n_hidden),
    Tanh(),
    # Linear(          n_hidden,n_hidden),BatchNormId(n_hidden),Tanh(),
    # Linear(          n_hidden,n_hidden),BatchNormId(n_hidden),Tanh(),
    # Linear(          n_hidden,n_hidden),BatchNormId(n_hidden),Tanh(),
    # Linear(          n_hidden,n_hidden),BatchNormId(n_hidden),Tanh(),
    Linear(n_hidden, n_hidden),
    BatchNormId(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
    BatchNormId(vocab_size),
]
# layers=[
#     Linear(n_embed*block_size,n_hidden),Tanh(),
#     # Linear(          n_hidden,n_hidden),BatchNormId(n_hidden),Tanh(),
#     # Linear(          n_hidden,n_hidden),BatchNormId(n_hidden),Tanh(),
#     # Linear(          n_hidden,n_hidden),BatchNormId(n_hidden),Tanh(),
#     # Linear(          n_hidden,n_hidden),BatchNormId(n_hidden),Tanh(),
#     Linear(          n_hidden,n_hidden),Tanh(),
#     Linear(          n_hidden,vocab_size),
# ]

with torch.no_grad():
    layers[-1].gamma *= 0.1
    # layers[-1].weight*=0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 1  # 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
param = 0
for p in parameters:
    param += p.nelement()
    p.requires_grad = True
print(param)
# Hyperparameters
lri = []
stepi = []
lossi = []
batch_size = 16
max_steps = 30000
lrs_e = [0.1, 0.01]
ud = []

for i in range(max_steps):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # Forward pass
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb)
    # print(loss.item())

    # Backward pass
    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update
    # lr=lrs[i]
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad

    # Track stats
    # lri.append(lre[i])
    if i % 10000 == 0:
        print(f"{1:7d}/{max_steps:7d}:{loss.item():.4f}")
    stepi.append(i)
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append(
            [((lr * p.grad).std() / p.data.std()).log10().item() for p in parameters]
        )
    if i >= 1000:
        break
print(loss.item())

# visualize histograms
plt.figure(figsize=(20, 4))  # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]):  # note: exclude the output layer
    if isinstance(layer, Tanh):
        t = layer.out
        print(
            "layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%"
            % (
                i,
                layer.__class__.__name__,
                t.mean(),
                t.std(),
                (t.abs() > 0.97).float().mean() * 100,
            )
        )
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer.__class__.__name__}")
plt.legend(legends)
plt.title("activation distribution")
# visualize histograms
plt.figure(figsize=(20, 4))  # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]):  # note: exclude the output layer
    if isinstance(layer, Tanh):
        t = layer.out.grad
        print(
            "layer %d (%10s): mean %+f, std %e"
            % (i, layer.__class__.__name__, t.mean(), t.std())
        )
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({layer.__class__.__name__}")
plt.legend(legends)
plt.title("gradient distribution")
# visualize histograms
plt.figure(figsize=(20, 4))  # width and height of the plot
legends = []
for i, p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2:
        print(
            "weight %10s | mean %+f | std %e | grad:data ratio %e"
            % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std())
        )
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"{i} {tuple(p.shape)}")
plt.legend(legends)
plt.title("weights gradient distribution")
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot([ud[j][i] for j in range(len(ud))])
        legends.append("param %d" % i)
plt.plot([0, len(ud)], [-3, -3], "k")  # these ratios should be ~1e-3, indicate on plot
plt.legend(legends)

import torch
import torch.nn.functional as F

words = open("names.txt", "r").read().splitlines()
alphabets = set("abcdefghijklmnopqrstuvwxyz")
words = [name for name in words if all(char in alphabets for char in name.lower())]
method = input("method p  for probability , n for neural network")
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
num = int(input("how many more do you want to generate ?"))
# print(num)
# num=min(num,300)
if method == "p":
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    P = N.float()
    P.sum(1, keepdim=True).shape
    P = P / P.sum(1, keepdim=True)
    g = torch.Generator().manual_seed(19)
    for i in range(num):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print("".join(out))
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
            # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

    # print(f'{log_likelihood=}')
    nll = -log_likelihood
    # print(f'{nll=}')
    loss = {nll / n}
    print("Loss = ", loss.item())

else:
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            # print(f'{ch1} {ch2}')
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num_element = xs.nelement()
    print("no. if examples:", num_element)

    g = torch.Generator().manual_seed(19)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    # gradient descent
    for k in range(100):
        # Forward pass
        xenc = F.one_hot(xs, num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = (
            -probs[torch.arange(num_element), ys].log().mean() + 0.0001 * (W**2).mean()
        )
        # print(loss.item())
        # Backward pass
        W.grad = None
        loss.backward()

        # Update
        W.data += -50 * W.grad

    # After
    g = torch.Generator().manual_seed(19)

    for i in range(num):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)
            # p=P[ix]
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print("".join(out))
        print("Loss = ", loss.item())

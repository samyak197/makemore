import math
import random
from micrograd_engine import Value
from micrograd_nn import Neuron, Layer, MLP

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, -1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
ypred = [n(x) for x in xs]
ypred
for k in range(200):
    # forward pass
    ys = [1.0, -1.0, -1.0, 1.0]
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05 * p.grad
    if k % 10 == 0:
        print(k, loss.data)

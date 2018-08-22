import matplotlib.pyplot as plt
import numpy as np

testbias = 1.0 * - 2.0

def sigmoid(x, bias=0):
    a = []
    for item in x:
        a.append(1/(1+np.exp(-(item+bias))))
    return a

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x, testbias)

plt.plot(x, sig)
plt.show()


def rectified_linear_unit(x):
    a = []
    for item in x:
        a.append(item * (item > 0))
    return a

x = np.arange(-5., 5., 0.2)
relu = rectified_linear_unit(x)

plt.plot(x, relu)
plt.show()


def soft_max(x):
    a = []
    for item in x:
        a.append(item * (item > 0))
    return a

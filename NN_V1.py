import numpy as np 

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e/e.sum()

def forward(X, W1, b1, W2, b2):
    z1 = W1@X + b1
    a1 = relu(z1)
    z2 = W2@a1 + b2
    out = softmax(z2)
    return (z1,a1,z2,out)


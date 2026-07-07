import numpy as np


class Embedding:
    """Token embedding layer"""

    def __init__(self, vocab_size, d_model):
        self.E = 0.02 * np.random.randn(vocab_size, d_model)
        self.dE = None
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return self.E[x]

    def backward(self, d_out):
        self.dE = np.zeros_like(self.E)
        B, T, D = d_out.shape
        for b in range(B):
            for t in range(T):
                self.dE[self.last_input[b, t]] += d_out[b, t]

    def update(self, lr):
        self.E -= lr * self.dE
        self.dE = None
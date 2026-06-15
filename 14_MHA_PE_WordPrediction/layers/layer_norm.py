import numpy as np

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((1, 1, dim))
        self.beta  = np.zeros((1, 1, dim))

        self.d_gamma = None
        self.d_beta  = None
        self.cache = None

    def forward(self, x):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)

        self.inv_std = 1.0 / np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mean) * self.inv_std

        self.cache = x
        return self.gamma * self.x_norm + self.beta

    def backward(self, d_out):
        B, T, D = d_out.shape
        x = self.cache

        self.d_gamma = np.sum(d_out * self.x_norm, axis=(0,1), keepdims=True)
        self.d_beta  = np.sum(d_out, axis=(0,1), keepdims=True)

        dx_norm = d_out * self.gamma

        dvar = np.sum(dx_norm * (x - self.mean) * -0.5 * self.inv_std**3, axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -self.inv_std, axis=-1, keepdims=True) + dvar * np.mean(-2*(x - self.mean), axis=-1, keepdims=True)

        dx = dx_norm * self.inv_std + dvar * 2*(x - self.mean)/D + dmean / D

        return dx

    def update(self, lr):
        self.gamma -= lr * self.d_gamma
        self.beta  -= lr * self.d_beta

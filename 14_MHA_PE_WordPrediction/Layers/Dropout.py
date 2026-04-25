import numpy as np

class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        return x * self.mask / (1 - self.p)

    def backward(self, d_out):
        if not self.training or self.p == 0:
            return d_out

        return d_out * self.mask / (1 - self.p)

    def update(self, lr):
        pass  # No params

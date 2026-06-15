import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len).reshape(-1,1)
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0)/d_model))
        self.pe[:,0::2] = np.sin(pos * div)
        self.pe[:,1::2] = np.cos(pos * div)

    def forward(self, x):
        B,T,D = x.shape
        return x + self.pe[:T]

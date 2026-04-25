import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        limit = np.sqrt(6/(input_dim+output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))

        self.dW = None
        self.db = None
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return x @ self.W + self.b

    def backward(self, d_out):
        B, T, D_in = self.last_input.shape
        x = self.last_input.reshape(B*T, D_in)
        d_out2 = d_out.reshape(B*T, -1)

        self.dW = x.T @ d_out2
        self.db = np.sum(d_out2, axis=0, keepdims=True)

        dx = d_out2 @ self.W.T
        return dx.reshape(B, T, D_in)

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

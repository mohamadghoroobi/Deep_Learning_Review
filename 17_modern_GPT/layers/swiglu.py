import numpy as np
from .dropout import Dropout

class SwiGLU:
    """
    SwiGLU Activation: Swish(x) * Gate(x)

    SwiGLU(x) = Swish(x @ W1) * (x @ W2)
    Where Swish(x) = x * sigmoid(x)

    Used in: LLaMA, PaLM, GPT-NeoX, Mistral
    """

    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: Input/Output dimension
            d_ff: Intermediate expansion dimension
        """
        # Information path
        self.W1 = np.random.randn(d_model, d_ff) * 0.01

        # Gate path
        self.W2 = np.random.randn(d_model, d_ff) * 0.01

        # Output projection
        self.W3 = np.random.randn(d_ff, d_model) * 0.01

        # Gradients
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)
        self.dW3 = np.zeros_like(self.W3)

        # Cache for backward
        self.cache = {}

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input (B, T, D)

        Returns:
            Output (B, T, D)
        """
        B, T, D = x.shape
        d_ff = self.W1.shape[1]

        # Information path: a = x @ W1
        a = x @ self.W1  # (B, T, d_ff)

        # Gate path: b = x @ W2
        b = x @ self.W2  # (B, T, d_ff)

        # Swish on a: a * sigmoid(a)
        sigmoid_a = 1 / (1 + np.exp(-a))
        swish_a = a * sigmoid_a

        # Gate with sigmoid(b)
        sigmoid_b = 1 / (1 + np.exp(-b))

        # Combine
        h = swish_a * sigmoid_b  # (B, T, d_ff)

        # Output projection
        y = h @ self.W3  # (B, T, D)

        # Cache for backward
        self.cache = {
            'x': x,
            'a': a,
            'b': b,
            'sigmoid_a': sigmoid_a,
            'swish_a': swish_a,
            'sigmoid_b': sigmoid_b,
            'h': h,
            'y': y
        }

        return y

    def backward(self, d_out):
        """Backward pass"""
        # Unpack cache
        x = self.cache['x']
        a = self.cache['a']
        b = self.cache['b']
        sigmoid_a = self.cache['sigmoid_a']
        swish_a = self.cache['swish_a']
        sigmoid_b = self.cache['sigmoid_b']
        h = self.cache['h']

        B, T, D = x.shape
        d_ff = self.W1.shape[1]

        # ----- Gradient through W3 -----
        h_flat = h.reshape(B * T, d_ff)
        d_out_flat = d_out.reshape(B * T, D)
        self.dW3 += h_flat.T @ d_out_flat

        # dh = d_out @ W3^T
        dh = d_out @ self.W3.T  # (B, T, d_ff)

        # ----- Gradient through combine -----
        # h = swish_a * sigmoid_b
        d_swish_a = dh * sigmoid_b
        d_sigmoid_b = dh * swish_a

        # ----- Gradient through sigmoid_b -----
        db = d_sigmoid_b * sigmoid_b * (1 - sigmoid_b)

        # ----- Gradient through W2 -----
        x_flat = x.reshape(B * T, D)
        db_flat = db.reshape(B * T, d_ff)
        self.dW2 += x_flat.T @ db_flat

        # dx from b path
        dx_b = db @ self.W2.T  # (B, T, D)

        # ----- Gradient through swish_a -----
        # swish_a = a * sigmoid(a)
        da = d_swish_a * (sigmoid_a + a * sigmoid_a * (1 - sigmoid_a))

        # ----- Gradient through W1 -----
        da_flat = da.reshape(B * T, d_ff)
        self.dW1 += x_flat.T @ da_flat

        # dx from a path
        dx_a = da @ self.W1.T  # (B, T, D)

        # Total dx
        dx = dx_a + dx_b

        return dx

    def update(self, lr):
        """Update weights"""
        self.W1 -= lr * self.dW1
        self.W2 -= lr * self.dW2
        self.W3 -= lr * self.dW3

        self.dW1.fill(0)
        self.dW2.fill(0)
        self.dW3.fill(0)


class SwiGLUFFN:
    """
    Complete Feed-Forward Network using SwiGLU

    This is what you'd use in a Transformer block.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        self.swiglu = SwiGLU(d_model, d_ff)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        h = self.swiglu.forward(x)
        y = self.dropout.forward(h)
        return y

    def backward(self, d_out):
        d_h = self.dropout.backward(d_out)
        dx = self.swiglu.backward(d_h)
        return dx

    def update(self, lr):
        self.swiglu.update(lr)
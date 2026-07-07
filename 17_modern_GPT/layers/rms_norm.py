import numpy as np

class RMSNorm:
    """
    Root Mean Square Layer Normalization

    Simpler and faster than LayerNorm.
    Used in: LLaMA, Mistral, all modern models.

    Key differences from LayerNorm:
    - No mean centering
    - No beta parameter
    - Only RMS scaling
    """

    def __init__(self, dim, eps=1e-5):
        """
        Args:
            dim: Feature dimension
            eps: Small constant for numerical stability
        """
        self.eps = eps
        self.gamma = np.ones((1, 1, dim))
        self.d_gamma = None
        self.cache = {}

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input (B, T, D)

        Returns:
            Normalized output (B, T, D)
        """
        # RMS = sqrt(mean(x²) + eps)
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)

        # Normalize: x / rms
        x_norm = x / rms

        # Scale: gamma * x_norm
        y = self.gamma * x_norm

        # Cache
        self.cache = {
            'x': x,
            'rms': rms,
            'x_norm': x_norm,
            'y': y
        }

        return y

    def backward(self, d_out):
        """Backward pass"""
        x = self.cache['x']
        rms = self.cache['rms']
        x_norm = self.cache['x_norm']

        B, T, D = d_out.shape

        # Gamma gradient
        self.d_gamma = np.sum(d_out * x_norm, axis=(0, 1), keepdims=True)

        # Input gradient
        d_x_norm = d_out * self.gamma
        d_x = d_x_norm / rms
        d_rms = -np.sum(d_x_norm * x, axis=-1, keepdims=True) / (rms**2 * D)
        d_mean_x2 = d_rms / (2 * rms)
        d_x_mean = d_mean_x2 * (2 * x) / D

        dx = d_x + d_x_mean

        return dx

    def update(self, lr):
        if self.d_gamma is not None:
            self.gamma -= lr * self.d_gamma
            self.d_gamma = None
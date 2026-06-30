import numpy as np
import matplotlib.pyplot as plt


class RotaryPositionalEmbedding:
    """
    Rotary Positional Embedding (RoPE)
    """

    def __init__(self, d_model, max_len=2048, base=10000.0):
        assert d_model % 2 == 0, "d_model must be even"

        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Pre-compute frequencies
        self.freqs = self._compute_frequencies()
        self._build_cache(max_len)

    def _compute_frequencies(self):
        d = np.arange(0, self.d_model // 2)
        freqs = self.base ** (-2.0 * d / self.d_model)
        return freqs

    def _build_cache(self, max_len):
        positions = np.arange(max_len)
        angles = np.outer(positions, self.freqs)
        self.cos_cache = np.cos(angles)
        self.sin_cache = np.sin(angles)

    def forward(self, x, start_pos=0):
        B, T, D = x.shape
        assert D == self.d_model

        cos = self.cos_cache[start_pos:start_pos + T]
        sin = self.sin_cache[start_pos:start_pos + T]

        cos = cos[None, :, :]
        sin = sin[None, :, :]

        # Split into pairs
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # Rotate
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos

        # Interleave back
        rotated = np.stack([x1_new, x2_new], axis=-1)
        rotated = rotated.reshape(B, T, D)

        return rotated
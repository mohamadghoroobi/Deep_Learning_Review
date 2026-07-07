import numpy as np


class RotaryPositionalEmbedding:
    """
    Rotary Positional Embedding (RoPE)

    Used in: LLaMA, GPT-NeoX, PaLM, Mistral

    Instead of adding position to embedding, we rotate the embedding
    by position-dependent angles. This naturally encodes relative positions.
    """

    def __init__(self, d_model, max_len=2048, base=10000.0):
        """
        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length
            base: Base for frequency calculation
        """
        assert d_model % 2 == 0, "d_model must be even"

        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Pre-compute frequencies for each dimension pair
        # freq_d = base^(-2d/d_model)
        self.freqs = self._compute_frequencies()

        # Pre-compute cos and sin for all positions
        self._build_cache(max_len)

    def _compute_frequencies(self):
        """Compute frequencies for each dimension pair"""
        d = np.arange(0, self.d_model // 2)
        freqs = self.base ** (-2.0 * d / self.d_model)
        return freqs

    def _build_cache(self, max_len):
        """Pre-compute cos and sin for all positions"""
        positions = np.arange(max_len)
        angles = np.outer(positions, self.freqs)
        self.cos_cache = np.cos(angles)  # (max_len, d_model/2)
        self.sin_cache = np.sin(angles)  # (max_len, d_model/2)

    def forward(self, x, start_pos=0):
        """
        Apply rotary positional embedding

        Args:
            x: Input tensor (B, T, D)
            start_pos: Starting position (for KV-cache)

        Returns:
            Rotated tensor (B, T, D)
        """
        B, T, D = x.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # Get cos and sin for positions
        cos = self.cos_cache[start_pos:start_pos + T]  # (T, D/2)
        sin = self.sin_cache[start_pos:start_pos + T]  # (T, D/2)

        # Expand to batch dimension
        cos = cos[None, :, :]  # (1, T, D/2)
        sin = sin[None, :, :]  # (1, T, D/2)

        # Split x into two halves
        x1 = x[..., 0::2]  # (B, T, D/2)
        x2 = x[..., 1::2]  # (B, T, D/2)

        # Apply rotation:
        # x1' = x1 * cos - x2 * sin
        # x2' = x1 * sin + x2 * cos
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos

        # Interleave back
        rotated = np.stack([x1_new, x2_new], axis=-1)
        rotated = rotated.reshape(B, T, D)

        return rotated

    def forward_with_cache(self, x, positions):
        """
        Apply RoPE with specific positions (for KV-cache)

        Args:
            x: Input tensor (B, 1, D) - single token
            positions: Position indices for each batch item

        Returns:
            Rotated tensor (B, 1, D)
        """
        B, T, D = x.shape
        assert T == 1, "Cache mode only supports single token"

        # Get cos and sin for specific positions
        cos = self.cos_cache[positions]  # (B, D/2)
        sin = self.sin_cache[positions]  # (B, D/2)

        # Add sequence dimension
        cos = cos[:, None, :]  # (B, 1, D/2)
        sin = sin[:, None, :]  # (B, 1, D/2)

        # Apply rotation
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos

        rotated = np.stack([x1_new, x2_new], axis=-1)
        rotated = rotated.reshape(B, T, D)

        return rotated
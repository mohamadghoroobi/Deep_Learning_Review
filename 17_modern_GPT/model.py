"""
Modern GPT Implementation with:
- RoPE (Rotary Positional Embedding)
- GQA (Grouped Query Attention)
- SwiGLU Activation
- Pre-Norm Architecture
- RMSNorm
"""

import numpy as np
from layers import (
    Embedding,
    RotaryPositionalEmbedding,
    GroupedQueryAttention,
    SwiGLUFFN,
    RMSNorm,
    Dropout,
    DenseLayer
)


class ModernDecoderBlock:
    """
    Modern Transformer Decoder Block

    Features:
    - Pre-Norm architecture
    - RMSNorm
    - GQA (Grouped Query Attention)
    - SwiGLU
    - Residual connections
    """

    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, dropout=0.1):
        # RMSNorm for Pre-Norm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # GQA attention (or standard if num_kv_heads=None)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads)

        # SwiGLU FFN
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

        # Dropout
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Cache for backward
        self.cache = {}

    def forward(self, x, mask=None):
        """
        Pre-Norm architecture:
        x = x + attention(norm(x))
        x = x + ffn(norm(x))
        """
        # Attention with Pre-Norm
        z = self.norm1.forward(x)
        att = self.attn.forward(z, z, z, mask)
        att = self.dropout1.forward(att)
        x = x + att

        # FFN with Pre-Norm
        z = self.norm2.forward(x)
        ff = self.ffn.forward(z)
        ff = self.dropout2.forward(ff)
        x = x + ff

        self.cache = {'x': x}
        return x

    def backward(self, d_out):
        d = d_out

        # FFN block
        d_ff = self.dropout2.backward(d)
        d_z2 = self.ffn.backward(d_ff)
        d_x2 = self.norm2.backward(d_z2)
        d = d + d_x2

        # Attention block
        d_att = self.dropout1.backward(d)
        d_q, d_k, d_v = self.attn.backward(d_att)
        d_z1 = d_q + d_k + d_v
        d_x1 = self.norm1.backward(d_z1)
        d = d + d_x1

        return d

    def update(self, lr):
        self.norm1.update(lr)
        self.norm2.update(lr)
        self.attn.update(lr)
        self.ffn.update(lr)


class ModernGPT:
    """
    Complete Modern GPT Model

    Architecture:
    - Embedding
    - RoPE
    - N × ModernDecoderBlock
    - LM Head (project to vocabulary)

    Features:
    - Pre-Norm
    - RMSNorm
    - GQA
    - SwiGLU
    - RoPE
    """

    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512,
                 num_layers=3, num_kv_heads=None, max_len=64, dropout=0.1):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: FFN expansion dimension
            num_layers: Number of decoder blocks
            num_kv_heads: Number of KV heads (GQA). If None, use MHA.
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Embedding
        self.embedding = Embedding(vocab_size, d_model)

        # RoPE
        self.rope = RotaryPositionalEmbedding(d_model, max_len)

        # Decoder blocks
        self.layers = [
            ModernDecoderBlock(d_model, num_heads, d_ff, num_kv_heads, dropout)
            for _ in range(num_layers)
        ]

        # Final RMSNorm
        self.norm = RMSNorm(d_model)

        # LM Head
        self.lm_head = DenseLayer(d_model, vocab_size)

        # Training mode
        self.training = True
        self._set_training_mode()

    def _set_training_mode(self):
        """Set dropout training mode for all layers"""
        for layer in self.layers:
            layer.dropout1.training = self.training
            layer.dropout2.training = self.training

    def forward(self, x, mask=None):
        """
        Forward pass

        Args:
            x: Input tokens (B, T)
            mask: Optional padding mask

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x.shape

        # Embedding
        h = self.embedding.forward(x)  # (B, T, D)

        # RoPE
        h = self.rope.forward(h)  # (B, T, D)

        # Decoder blocks
        for layer in self.layers:
            h = layer.forward(h, mask)

        # Final RMSNorm
        h = self.norm.forward(h)

        # LM Head
        logits = self.lm_head.forward(h)  # (B, T, V)

        return logits

    def backward(self, d_logits):
        """Backward pass"""
        # LM Head
        d_h = self.lm_head.backward(d_logits)

        # Final RMSNorm
        d_h = self.norm.backward(d_h)

        # Decoder blocks (reverse order)
        for layer in reversed(self.layers):
            d_h = layer.backward(d_h)

        # Embedding
        self.embedding.backward(d_h)

    def update(self, lr):
        """Update all weights"""
        self.embedding.update(lr)
        for layer in self.layers:
            layer.update(lr)
        self.norm.update(lr)
        self.lm_head.update(lr)

    def train_mode(self):
        """Set model to training mode"""
        self.training = True
        self._set_training_mode()

    def eval_mode(self):
        """Set model to evaluation mode"""
        self.training = False
        self._set_training_mode()
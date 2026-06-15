import numpy as np

from layers.embedding import Embedding
from layers.positional_encoding import PositionalEncoding
from layers.multi_head_attention import MultiHeadAttention
from layers.dense_layer import DenseLayer
from layers.layer_norm import LayerNorm
from layers.dropout import Dropout
from layers.masks import create_causal_mask


class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff1 = DenseLayer(d_model, d_ff)  # 128 → 512
        self.ff2 = DenseLayer(d_ff, d_model)  # 512 → 128

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Cache for ReLU gradient
        self.relu_input = None  # Will store ff1 output before ReLU

    def forward(self, x):
        # ===== SELF ATTENTION =====
        z = self.ln1.forward(x)
        B, T, _ = z.shape
        mask = create_causal_mask(T).reshape(1, 1, T, T)

        att = self.self_attn.forward(z, z, z, mask)
        att = self.dropout1.forward(att)
        x = x + att

        # ===== FEED FORWARD =====
        z2 = self.ln2.forward(x)

        # FF1: Expand dimension
        ff = self.ff1.forward(z2)  # (B, T, 128) → (B, T, 512)

        # Cache for ReLU backward
        self.relu_input = ff.copy()  # ← SAVE BEFORE ReLU

        # ReLU activation
        ff = np.maximum(ff, 0)  # (B, T, 512)

        # FF2: Project back
        ff = self.ff2.forward(ff)  # (B, T, 512) → (B, T, 128)

        ff = self.dropout2.forward(ff)
        x = x + ff

        return x

    def backward(self, d_out):
        d = d_out

        # ===== FEED FORWARD BACKWARD =====
        # Backward through dropout2
        dz = self.dropout2.backward(d)  # (B, T, 128)

        # Backward through ff2
        dz = self.ff2.backward(dz)  # (B, T, 512)

        # Backward through ReLU (using cached input)
        dz_relu = dz * (self.relu_input > 0)  # ← FIXED: now shapes match!

        # Backward through ff1
        dz = self.ff1.backward(dz_relu)  # (B, T, 128)

        # Backward through layer norm
        dz = self.ln2.backward(dz)  # (B, T, 128)

        # Add residual gradient
        d = d + dz

        # ===== SELF ATTENTION BACKWARD =====
        # Backward through dropout1
        da = self.dropout1.backward(d)  # (B, T, 128)

        # Backward through self attention
        da = self.self_attn.backward(da)  # (B, T, 128)

        # Backward through layer norm
        da = self.ln1.backward(da)  # (B, T, 128)

        # Add residual gradient
        d = d + da

        return d

    def update(self, lr):
        self.self_attn.update(lr)
        self.ff1.update(lr)
        self.ff2.update(lr)
        self.ln1.update(lr)
        self.ln2.update(lr)


class TransformerLM:
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512, num_Layers=2, max_len=256, dropout=0.1):
        self.embedding = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)

        self.Layers = [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_Layers)]
        self.out = DenseLayer(d_model, vocab_size)

        # Set training mode
        self.training = True
        for layer in self.Layers:
            layer.dropout1.training = True
            layer.dropout2.training = True

    def forward(self, x):
        h = self.embedding.forward(x)
        h = self.pe.forward(h)
        for layer in self.Layers:
            h = layer.forward(h)
        return self.out.forward(h)

    def backward(self, d_logits):
        d = self.out.backward(d_logits)
        for layer in reversed(self.Layers):
            d = layer.backward(d)
        self.embedding.backward(d)

    def update(self, lr):
        self.out.update(lr)
        for l in self.Layers:
            l.update(lr)
        self.embedding.update(lr)

    def eval(self):
        """Set model to evaluation mode (disables dropout)"""
        self.training = False
        for layer in self.Layers:
            layer.dropout1.training = False
            layer.dropout2.training = False

    def train(self):
        """Set model to training mode (enables dropout)"""
        self.training = True
        for layer in self.Layers:
            layer.dropout1.training = True
            layer.dropout2.training = True
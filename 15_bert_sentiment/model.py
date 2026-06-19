import numpy as np
from layers.embedding import Embedding
from layers.positional_encoding import PositionalEncoding
from layers.multi_head_attention import MultiHeadAttention
from layers.dense_layer import DenseLayer, DenseLayer2D
from layers.layer_norm import LayerNorm
from layers.dropout import Dropout


class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff1 = DenseLayer(d_model, d_ff)
        self.ff2 = DenseLayer(d_ff, d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Cache for ReLU gradient
        self.relu_input = None

    def forward(self, x, mask=None):
        # ===== SELF-ATTENTION (NO CAUSAL MASK) =====
        z = self.ln1.forward(x)
        att = self.self_attn.forward(z, z, z, mask)
        att = self.dropout1.forward(att)
        x = x + att

        # ===== FEED FORWARD =====
        z2 = self.ln2.forward(x)
        ff = self.ff1.forward(z2)
        self.relu_input = ff.copy()
        ff = np.maximum(ff, 0)
        ff = self.ff2.forward(ff)
        ff = self.dropout2.forward(ff)
        x = x + ff

        return x

    def backward(self, d_out):
        d = d_out

        # FF block
        dz = self.dropout2.backward(d)
        dz = self.ff2.backward(dz)
        dz_relu = dz * (self.relu_input > 0)
        dz = self.ff1.backward(dz_relu)
        dz = self.ln2.backward(dz)
        d = d + dz

        # Attention block
        da = self.dropout1.backward(d)
        da_q, da_k, da_v = self.self_attn.backward(da)
        da = da_q + da_k + da_v
        da = self.ln1.backward(da)
        d = d + da

        return d

    def update(self, lr):
        self.self_attn.update(lr)
        self.ff1.update(lr)
        self.ff2.update(lr)
        self.ln1.update(lr)
        self.ln2.update(lr)


class BERTEncoder:
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512,
                 num_layers=3, max_len=128, dropout=0.1):
        self.embedding = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)

        self.layers = [EncoderBlock(d_model, num_heads, d_ff, dropout)
                       for _ in range(num_layers)]

        # Pooler and Classifier use 2D Dense Layer
        self.pooler = DenseLayer2D(d_model, d_model)
        self.classifier = DenseLayer2D(d_model, 2)

        # Store last layer output for backward
        self.last_hidden = None
        self.last_x = None

    def forward(self, x, mask=None):
        # x: (B, T) token ids
        self.last_x = x.copy()

        h = self.embedding.forward(x)
        h = self.pe.forward(h)

        for layer in self.layers:
            h = layer.forward(h, mask)

        self.last_hidden = h.copy()

        # Take [CLS] token (first position) for classification
        cls_token = h[:, 0, :]  # (B, d_model)

        # Pooler
        pooled = self.pooler.forward(cls_token)
        pooled = np.tanh(pooled)

        # Classification
        logits = self.classifier.forward(pooled)

        return logits, h

    def backward(self, d_logits):
        # Backward through classifier
        d_pooled = self.classifier.backward(d_logits)

        # Backward through tanh
        d_pooled = d_pooled * (1 - np.tanh(self.pooler.last_input) ** 2)

        # Backward through pooler
        d_cls = self.pooler.backward(d_pooled)

        # Simplified: only propagate gradient through [CLS] token
        d_h = np.zeros_like(self.last_hidden)
        d_h[:, 0, :] = d_cls

        # Backward through layers
        for layer in reversed(self.layers):
            d_h = layer.backward(d_h)

        # Backward through embeddings
        self.embedding.backward(d_h)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)
        self.pooler.update(lr)
        self.classifier.update(lr)
        self.embedding.update(lr)
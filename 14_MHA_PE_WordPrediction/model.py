import numpy as np

from Layers.Embedding import Embedding
from Layers.PositionalEncoding import PositionalEncoding
from Layers.MultiHeadAttention import MultiHeadAttention
from Layers.DenseLayer import DenseLayer
from Layers.LayerNorm import LayerNorm
from Layers.Dropout import Dropout
from Layers.Masks import create_causal_mask


class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)



        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff1 = DenseLayer(d_model, d_ff)
        self.ff2 = DenseLayer(d_ff, d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x):
        # Self attention
        z = self.ln1.forward(x)
        B,T,_ = z.shape
        mask = create_causal_mask(T).reshape(1,1,T,T)

        att = self.self_attn.forward(z, z, z, mask)
        att = self.dropout1.forward(att)
        x = x + att

        # Feed forward
        z2 = self.ln2.forward(x)
        ff = self.ff1.forward(z2)
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
        dz_relu = dz * (self.ff1.last_input > 0)
        dz = self.ff1.backward(dz_relu)
        dz = self.ln2.backward(dz)

        d = d + dz

        # Attention block
        da = self.dropout1.backward(d)
        da = self.self_attn.backward(da)
        da = self.ln1.backward(da)

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

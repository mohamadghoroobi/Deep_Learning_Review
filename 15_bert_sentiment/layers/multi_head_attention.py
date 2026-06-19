import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        k = np.sqrt(1 / d_model)

        self.Wq = np.random.uniform(-k, k, (d_model, d_model))
        self.Wk = np.random.uniform(-k, k, (d_model, d_model))
        self.Wv = np.random.uniform(-k, k, (d_model, d_model))
        self.Wo = np.random.uniform(-k, k, (d_model, d_model))

        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dWo = np.zeros_like(self.Wo)

        self.cache = None

    def split_heads(self, x):
        B, T, D = x.shape
        return x.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    def forward(self, q, k, v, mask=None):
        self.q_in, self.k_in, self.v_in = q, k, v

        Q = q @ self.Wq
        K = k @ self.Wk
        V = v @ self.Wv

        self.Q = self.split_heads(Q)
        self.K = self.split_heads(K)
        self.V = self.split_heads(V)

        att = self.Q @ self.K.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)

        if mask is not None:
            att = att + mask * -1e9

        self.att_soft = np.exp(att - att.max(axis=-1, keepdims=True))
        self.att_soft /= np.sum(self.att_soft, axis=-1, keepdims=True)

        out = self.att_soft @ self.V
        out = self.combine_heads(out)
        self.out_before_proj = out

        return out @ self.Wo

    def backward(self, d_out):
        B, T, D = d_out.shape

        # Wo gradient
        dWo = self.out_before_proj.reshape(B * T, D).T @ d_out.reshape(B * T, D)
        self.dWo += dWo
        d_proj = d_out @ self.Wo.T

        d_proj_heads = self.split_heads(d_proj)

        # Attention gradient
        datt = d_proj_heads @ self.V.transpose(0, 1, 3, 2)
        dV = self.att_soft.transpose(0, 1, 3, 2) @ d_proj_heads
        dV = self.combine_heads(dV)

        Q = self.Q
        K = self.K

        d_att_weights = self.att_soft * (datt - np.sum(datt * self.att_soft, axis=-1, keepdims=True))

        dQ = d_att_weights @ K
        dK = d_att_weights.transpose(0, 1, 3, 2) @ Q

        dQ = self.combine_heads(dQ)
        dK = self.combine_heads(dK)
        dV = dV

        # Projection gradients
        self.dWq += self.q_in.reshape(B * T, -1).T @ dQ.reshape(B * T, -1)
        self.dWk += self.k_in.reshape(B * T, -1).T @ dK.reshape(B * T, -1)
        self.dWv += self.v_in.reshape(B * T, -1).T @ dV.reshape(B * T, -1)

        # Input gradients
        dq = dQ.reshape(B * T, -1) @ self.Wq.T
        dk = dK.reshape(B * T, -1) @ self.Wk.T
        dv = dV.reshape(B * T, -1) @ self.Wv.T

        return dq.reshape(B, T, D), dk.reshape(B, T, D), dv.reshape(B, T, D)

    def update(self, lr):
        self.Wq -= lr * self.dWq
        self.Wk -= lr * self.dWk
        self.Wv -= lr * self.dWv
        self.Wo -= lr * self.dWo

        self.dWq.fill(0)
        self.dWk.fill(0)
        self.dWv.fill(0)
        self.dWo.fill(0)
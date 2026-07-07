import numpy as np

class GroupedQueryAttention:
    """
    Grouped Query Attention (GQA)
    Used in: LLaMA 2, Gemini, PaLM 2
    """

    def __init__(self, d_model, num_heads, num_kv_heads):
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads

        # Query projections: each head projects to d_head
        self.Wq = np.random.randn(num_heads, d_model, self.d_head) * 0.01
        # Key/Value projections: each KV head projects to d_head
        self.Wk = np.random.randn(num_kv_heads, d_model, self.d_head) * 0.01
        self.Wv = np.random.randn(num_kv_heads, d_model, self.d_head) * 0.01
        # Output projection: from d_model to d_model
        self.Wo = np.random.randn(d_model, d_model) * 0.01

        # Gradients
        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dWo = np.zeros_like(self.Wo)

        self.cache = {}

    def forward(self, q, k, v, mask=None):
        B, T, D = q.shape
        H = self.num_heads
        KV = self.num_kv_heads
        Dh = self.d_head

        # ----- Projections -----
        # Q: (B, T, H, Dh)
        Q = np.einsum('btd,hde->bthe', q, self.Wq)
        # K: (B, T, KV, Dh)
        K = np.einsum('btd,hde->bthe', k, self.Wk)
        # V: (B, T, KV, Dh)
        V = np.einsum('btd,hde->bthe', v, self.Wv)

        # ----- Expand K and V to match number of heads -----
        K_expanded = np.repeat(K, self.num_groups, axis=2)
        V_expanded = np.repeat(V, self.num_groups, axis=2)

        # ----- Transpose to (B, H, T, Dh) -----
        Q = Q.transpose(0, 2, 1, 3)
        K_expanded = K_expanded.transpose(0, 2, 1, 3)
        V_expanded = V_expanded.transpose(0, 2, 1, 3)

        # ----- Compute attention scores -----
        scores = np.matmul(Q, K_expanded.transpose(0, 1, 3, 2)) / np.sqrt(Dh)

        if mask is not None:
            scores = scores + mask * -1e9

        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)

        # ----- Apply attention to values -----
        out = np.matmul(attn, V_expanded)

        # ----- Combine heads -----
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        # ----- Output projection -----
        out = out @ self.Wo

        # Cache for backward
        self.cache = {
            'q': q,
            'k': k,
            'v': v,
            'Q': Q,
            'K_expanded': K_expanded,
            'V_expanded': V_expanded,
            'attn': attn,
            'scores': scores,
            'out': out
        }

        return out

    def backward(self, d_out):
        B, T, D = d_out.shape
        H = self.num_heads
        KV = self.num_kv_heads
        Dh = self.d_head

        # Retrieve from cache
        q = self.cache['q']
        Q = self.cache['Q']
        K_expanded = self.cache['K_expanded']
        V_expanded = self.cache['V_expanded']
        attn = self.cache['attn']

        # ----- Gradient through Wo -----
        out = self.cache['out']
        self.dWo += out.reshape(B * T, D).T @ d_out.reshape(B * T, D)

        # ----- Gradient through output projection -----
        d_out = d_out @ self.Wo.T
        d_out = d_out.reshape(B, T, D)

        # ----- Reshape to heads -----
        d_out_heads = d_out.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        # Gradient through attention
        d_attn = np.matmul(d_out_heads, V_expanded.transpose(0, 1, 3, 2))
        dV_expanded = np.matmul(attn.transpose(0, 1, 3, 2), d_out_heads)

        # Gradient through softmax
        d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))

        # Gradient through scores
        dQ = np.matmul(d_scores, K_expanded)
        dK_expanded = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        # ----- Reduce K and V gradients to KV heads -----
        dK = dK_expanded.reshape(B, H, T, Dh)
        dK = dK.reshape(B, KV, self.num_groups, T, Dh).sum(axis=2)

        dV = dV_expanded.reshape(B, H, T, Dh)
        dV = dV.reshape(B, KV, self.num_groups, T, Dh).sum(axis=2)

        # ----- Projection gradients -----
        q_flat = q.reshape(B * T, D)
        dQ_flat = dQ.reshape(B * T, -1)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B * T, -1)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B * T, -1)

        # Per-head gradients
        for h in range(H):
            self.dWq[h] += q_flat.T @ dQ_flat.reshape(B * T, H, Dh)[:, h, :]

        for kv in range(KV):
            self.dWk[kv] += q_flat.T @ dK_flat.reshape(B * T, KV, Dh)[:, kv, :]
            self.dWv[kv] += q_flat.T @ dV_flat.reshape(B * T, KV, Dh)[:, kv, :]

        # ----- Input gradients -----
        dq = np.zeros_like(q)
        dk = np.zeros_like(q)
        dv = np.zeros_like(q)

        for h in range(H):
            # Compute (B*T, Dh) @ (Dh, D) -> (B*T, D), then reshape to (B, T, D)
            grad_q = dQ_flat.reshape(B * T, H, Dh)[:, h, :] @ self.Wq[h].T
            dq += grad_q.reshape(B, T, D)

        for kv in range(KV):
            grad_k = dK_flat.reshape(B * T, KV, Dh)[:, kv, :] @ self.Wk[kv].T
            grad_v = dV_flat.reshape(B * T, KV, Dh)[:, kv, :] @ self.Wv[kv].T
            dk += grad_k.reshape(B, T, D)
            dv += grad_v.reshape(B, T, D)

        return dq, dk, dv

    def update(self, lr):
        self.Wq -= lr * self.dWq
        self.Wk -= lr * self.dWk
        self.Wv -= lr * self.dWv
        self.Wo -= lr * self.dWo

        self.dWq.fill(0)
        self.dWk.fill(0)
        self.dWv.fill(0)
        self.dWo.fill(0)

    def get_kv_cache_size(self, batch, seq_len):
        return 2 * batch * seq_len * self.num_kv_heads * self.d_head
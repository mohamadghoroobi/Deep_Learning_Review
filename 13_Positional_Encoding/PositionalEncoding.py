import numpy as np


class Layer_Dense:

    def __init__(self, in_dim, out_dim):

        self.weights = 0.01*np.random.randn(in_dim, out_dim)
        self.biases = np.zeros(out_dim)

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, inputs):

        self.inputs = inputs              # (B,T,D)
        return inputs @ self.weights + self.biases

    def backward(self, dvalues):

        B,T,D = self.inputs.shape

        x = self.inputs.reshape(B*T, D)
        dy = dvalues.reshape(B*T, -1)

        self.dweights += x.T @ dy
        self.dbiases += dy.sum(axis=0)

        dinputs = dy @ self.weights.T
        return dinputs.reshape(B,T,D)

    def update(self, lr):

        self.weights -= lr*self.dweights
        self.biases -= lr*self.dbiases

        self.dweights[:] = 0
        self.dbiases[:] = 0


class Embedding:

    def __init__(self,vocab_size,d_model):

        self.weights = 0.01*np.random.randn(vocab_size,d_model)
        self.dweights = np.zeros_like(self.weights)

    def forward(self,idx):

        self.idx = idx                 # (B,T)
        return self.weights[idx]       # (B,T,D)

    def backward(self,dvalues):

        B,T,D = dvalues.shape

        for b in range(B):
            for t in range(T):
                token = self.idx[b,t]
                self.dweights[token] += dvalues[b,t]

    def update(self,lr):

        self.weights -= lr*self.dweights
        self.dweights[:] = 0


class PositionalEncoding:

    def __init__(self,d_model,max_len=5000):

        pe = np.zeros((max_len,d_model))

        pos = np.arange(max_len).reshape(-1,1)

        div = np.exp(np.arange(0,d_model,2)*(-np.log(10000)/d_model))

        pe[:,0::2] = np.sin(pos*div)
        pe[:,1::2] = np.cos(pos*div)

        self.pe = pe

    def forward(self,x):

        B,T,D = x.shape

        return x + self.pe[:T]



class MultiHeadAttention:

    def __init__(self,d_model,num_heads):

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model//num_heads

        self.Wq = np.random.randn(d_model,d_model)*0.01
        self.Wk = np.random.randn(d_model,d_model)*0.01
        self.Wv = np.random.randn(d_model,d_model)*0.01
        self.Wo = np.random.randn(d_model,d_model)*0.01

        self.dWq = np.zeros_like(self.Wq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dWo = np.zeros_like(self.Wo)

    def split_heads(self,x):

        B,T,D = x.shape
        x = x.reshape(B,T,self.num_heads,self.depth)
        return np.transpose(x,(0,2,1,3))

    def combine_heads(self,x):

        B,H,T,Dh = x.shape
        x = np.transpose(x,(0,2,1,3))
        return x.reshape(B,T,self.d_model)

    def forward(self,Q_in,K_in,V_in,mask=None):

        self.Q_in = Q_in
        self.K_in = K_in
        self.V_in = V_in

        self.Q = Q_in @ self.Wq
        self.K = K_in @ self.Wk
        self.V = V_in @ self.Wv

        Q = self.split_heads(self.Q)
        K = self.split_heads(self.K)
        V = self.split_heads(self.V)

        scores = Q @ np.transpose(K,(0,1,3,2))
        scores /= np.sqrt(self.depth)

        if mask is not None:
            scores += mask*-1e9

        self.attn = softmax(scores)

        context = self.attn @ V

        self.context = self.combine_heads(context)

        output = self.context @ self.Wo

        return output

    def backward(self, dvalues):

        B, Tq, D = self.Q_in.shape
        _, Tk, _ = self.K_in.shape
        _, Tv, _ = self.V_in.shape

        H = self.num_heads
        Dh = self.depth

        # ---- Wo ----
        context_flat = self.context.reshape(B * Tq, D)
        dvalues_flat = dvalues.reshape(B * Tq, D)

        self.dWo += context_flat.T @ dvalues_flat

        dcontext = dvalues @ self.Wo.T  # (B,T,D)

        # ---- split heads ----
        dcontext = self.split_heads(dcontext)  # (B,H,T,Dh)

        Qh = self.split_heads(self.Q)
        Kh = self.split_heads(self.K)
        Vh = self.split_heads(self.V)

        dQh = np.zeros_like(Qh)
        dKh = np.zeros_like(Kh)
        dVh = np.zeros_like(Vh)

        scale = np.sqrt(self.depth)

        for b in range(B):
            for h in range(H):
                attn = self.attn[b, h]  # (Tq, Tk)

                dcontext_h = dcontext[b, h]  # (Tq,Dh)

                # context = attn @ V
                dattn = dcontext_h @ Vh[b, h].T
                dVh[b, h] = attn.T @ dcontext_h

                # softmax backward
                ds = dattn * attn
                sum_ds = np.sum(ds, axis=-1, keepdims=True)
                dscores = ds - attn * sum_ds

                dscores /= scale

                # scores = QKᵀ
                dQh[b, h] = dscores @ Kh[b, h]
                dKh[b, h] = dscores.T @ Qh[b, h]

        # combine heads
        dQ = self.combine_heads(dQh)  # (B,T,D)
        dK = self.combine_heads(dKh)
        dV = self.combine_heads(dVh)

        # ---- projection gradients ----

        Q_flat = self.Q_in.reshape(B * Tq, D)
        K_flat = self.K_in.reshape(B * Tk, D)
        V_flat = self.V_in.reshape(B * Tk, D)

        dQ_flat = dQ.reshape(B * Tq, D)
        dK_flat = dK.reshape(B * Tk, D)
        dV_flat = dV.reshape(B * Tk, D)

        self.dWq += Q_flat.T @ dQ_flat
        self.dWk += K_flat.T @ dK_flat
        self.dWv += V_flat.T @ dV_flat

        # ---- input gradients ----

        dQ_in = dQ @ self.Wq.T
        dK_in = dK @ self.Wk.T
        dV_in = dV @ self.Wv.T

        return dQ_in, dK_in, dV_in

    def update(self, lr):
        self.Wq -= lr * self.dWq
        self.Wk -= lr * self.dWk
        self.Wv -= lr * self.dWv
        self.Wo -= lr * self.dWo

        self.dWq[:] = 0
        self.dWk[:] = 0
        self.dWv[:] = 0
        self.dWo[:] = 0

def create_causal_mask(T):

    mask = np.triu(np.ones((T,T)),k=1)
    return mask.reshape(1,1,T,T)


def softmax(x):
    # x: (..., V)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def softmax_cross_entropy(logits, target, mask=None):
    """
    logits: (B, T, V)
    target: (B, T) int indices
    mask:   (B, T) float {0,1}, 1 = keep, 0 = ignore (e.g. PAD). If None, no masking.
    """
    probs = softmax(logits)           # (B,T,V)

    B, T, V = probs.shape

    loss = 0.0
    grad = probs.copy()               # (B,T,V)

    if mask is None:
        # No masking: average over all positions
        denom = B * T
        for b in range(B):
            for t in range(T):
                y = target[b, t]
                loss -= np.log(probs[b, t, y] + 1e-9)
                grad[b, t, y] -= 1.0

        loss /= denom
        grad /= denom
        return loss, grad

    # With masking: average only over positions where mask == 1
    # mask shape: (B,T)
    mask = mask.astype(float)
    denom = np.sum(mask)  # number of valid (non-PAD) positions

    if denom == 0:
        # Edge case: everything is PAD, avoid divide-by-zero
        return 0.0, np.zeros_like(logits)

    for b in range(B):
        for t in range(T):
            if mask[b, t] == 0.0:
                # ignore PAD positions: zero grad
                grad[b, t, :] = 0.0
                continue

            y = target[b, t]
            loss -= np.log(probs[b, t, y] + 1e-9)
            grad[b, t, y] -= 1.0

    loss /= denom
    grad /= denom

    return loss, grad


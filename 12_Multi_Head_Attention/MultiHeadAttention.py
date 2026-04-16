import numpy as np


class MultiHeadAttention:
    """
    Multi-head self-attention with full forward and backward pass (NumPy).
    Input: H (T, d_model)
    """

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # projection matrices
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        # gradient buffers
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

    # -------------------------------------------------------------
    # splitting + combining
    # -------------------------------------------------------------
    def split_heads(self, X):
        # (T, d_model) -> (num_heads, T, head_dim)
        T = X.shape[0]
        X = X.reshape(T, self.num_heads, self.head_dim)
        return X.transpose(1, 0, 2)

    def combine_heads(self, X):
        # (num_heads, T, head_dim) -> (T, d_model)
        X = X.transpose(1, 0, 2)
        T = X.shape[0]
        return X.reshape(T, self.d_model)

    # -------------------------------------------------------------
    # forward
    # -------------------------------------------------------------
    def forward(self, H):
        """
        H: (T, d_model)
        returns:
            output: (T, d_model)
            alpha:  (num_heads, T, T)
        """
        self.H = H
        T = H.shape[0]

        # projections
        Q = H @ self.W_q
        K = H @ self.W_k
        V = H @ self.W_v

        # split into heads
        self.Q = self.split_heads(Q)    # (h, T, d_k)
        self.K = self.split_heads(K)
        self.V = self.split_heads(V)

        dk = np.sqrt(self.head_dim)

        self.alpha = []
        self.scores = []
        self.context_heads = []

        # compute head-wise attention
        for h in range(self.num_heads):
            Qh = self.Q[h]             # (T, d_k)
            Kh = self.K[h]
            Vh = self.V[h]

            scores = Qh @ Kh.T / dk
            scores = scores - np.max(scores, axis=-1, keepdims=True)

            exp_s = np.exp(scores)
            alpha = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

            context = alpha @ Vh

            self.scores.append(scores)
            self.alpha.append(alpha)
            self.context_heads.append(context)

        self.scores = np.stack(self.scores)
        self.alpha = np.stack(self.alpha)
        self.context_heads = np.stack(self.context_heads)

        # merge heads
        context = self.combine_heads(self.context_heads)
        self.context = context

        # final projection
        output = context @ self.W_o
        self.output = output

        return output, self.alpha

    # -------------------------------------------------------------
    # backward
    # -------------------------------------------------------------
    def backward(self, d_out):
        """
        d_out: gradient wrt final output (T, d_model)
        returns dH: gradient wrt input (T, d_model)
        """

        # --- dW_o and gradient into context ---
        self.dW_o = self.context.T @ d_out
        d_context = d_out @ self.W_o.T    # (T, d_model)

        # reshape to head-format (h, T, d_k)
        d_context_heads = self.split_heads(d_context)

        # init grads
        dQ = np.zeros_like(self.Q)
        dK = np.zeros_like(self.K)
        dV = np.zeros_like(self.V)

        dk = np.sqrt(self.head_dim)

        # --- loop over heads ---
        for h in range(self.num_heads):

            alpha = self.alpha[h]
            scores = self.scores[h]

            d_context_h = d_context_heads[h]  # (T, d_k)
            Vh = self.V[h]
            Kh = self.K[h]
            Qh = self.Q[h]

            # context = alpha @ Vh
            d_alpha = d_context_h @ Vh.T      # (T, T)
            dV[h] = alpha.T @ d_context_h     # (T, d_k)

            # softmax backward: d_alpha -> d_scores
            # s = softmax(scores)
            ds = d_alpha * alpha
            sum_ds = np.sum(ds, axis=-1, keepdims=True)
            d_scores = ds - alpha * sum_ds    # (T, T)

            # scores = Qh @ Kh.T / dk
            d_scores /= dk

            # dQh: (T, d_k)
            dQ[h] = d_scores @ Kh

            # dKh:
            dK[h] = d_scores.T @ Qh

        # combine heads into full Q,K,V gradient shape (T, d_model)
        dQ_full = self.combine_heads(dQ)
        dK_full = self.combine_heads(dK)
        dV_full = self.combine_heads(dV)

        # gradients w.r.t. projection matrices
        self.dW_q = self.H.T @ dQ_full
        self.dW_k = self.H.T @ dK_full
        self.dW_v = self.H.T @ dV_full

        # gradient wrt input
        dH_q = dQ_full @ self.W_q.T
        dH_k = dK_full @ self.W_k.T
        dH_v = dV_full @ self.W_v.T

        return dH_q + dH_k + dH_v


class Embedding:
    def __init__(self, vocab_size, d_model):
        self.W = np.random.randn(vocab_size, d_model) * 0.02

        # gradient
        self.dW = np.zeros_like(self.W)

    def forward(self, x_onehot):
        # x_onehot: (T, vocab_size)
        # output: (T, d_model)
        self.x = x_onehot
        return x_onehot @ self.W

    def backward(self, d_out):
        # d_out: (T, d_model)
        # dW = x^T @ d_out
        self.dW = self.x.T @ d_out
        # No gradient to x_onehot (discrete)
        return None


# --- Simple Dense Layer for Classification ---
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Weights and biases, initialized small
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Gradients
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, inputs):
        """
        inputs: (T, n_inputs) or (batch_size, n_inputs)
        returns: (T, n_neurons) or (batch_size, n_neurons)
        """
        self.inputs = inputs  # store for backward
        return inputs @ self.weights + self.biases

    def backward(self, dvalues):
        """
        dvalues: gradient of loss w.r.t. layer output, same shape as forward output
        """
        self.dweights = self.inputs.T @ dvalues              # (n_inputs, n_neurons)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dvalues @ self.weights.T
        return self.dinputs


# --- Simple activations & losses (optional helpers) ---

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    # derivative wrt x
    t = np.tanh(x)
    return 1.0 - t ** 2

def sigmoid(x):
    # numerically stable sigmoid
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def binary_cross_entropy(pred, target):
    """
    pred: (batch, 1) sigmoid outputs in (0, 1)
    target: (batch, 1) or (batch,) with 0/1
    """
    eps = 1e-12
    pred = np.clip(pred, eps, 1 - eps)
    target = target.reshape(pred.shape)
    loss = - (target * np.log(pred) + (1 - target) * np.log(1 - pred))
    return np.mean(loss)

def binary_cross_entropy_backward(pred, target):
    """
    derivative dL/d(pred)
    pred: (batch, 1)
    target: (batch, 1) or (batch,)
    """
    eps = 1e-12
    pred = np.clip(pred, eps, 1 - eps)
    target = target.reshape(pred.shape)
    # mean over batch, so divide by batch_size
    batch_size = pred.shape[0]
    return (-(target / pred) + (1 - target) / (1 - pred)) / batch_size
import numpy as np

# --- Core Self-Attention Layer ---
class SelfAttention:
    """
    Single-head self-attention layer (dot-product style).
    Works with sequence inputs: H shape (T, d_model)
    """
    def __init__(self, d_model):
        self.d_model = d_model
        # Initialize weights for Q, K, V, and Output projection
        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        # gradient buffers
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

    def forward(self, H):
        """
        H : (T, d_model) — sequence of token embeddings or hidden states
        returns:
            output : (T, d_model) — sequence of context-aware representations
            attn_weights : (T, T) attention matrix
        """
        self.H = H  # (T, d_model)
        T, d_model = H.shape

        # Project H into Q, K, V
        self.Q = H @ self.W_q           # (T, d_model)
        self.K = H @ self.W_k           # (T, d_model)
        self.V = H @ self.W_v           # (T, d_model)

        dk = np.sqrt(self.d_model)
        # Attention scores: Q @ K.T
        scores = self.Q @ self.K.T / dk      # (T, T)

        # Softmax to get attention weights (row-wise)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        self.alpha = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (T, T)

        # Compute context vectors: weighted sum of V
        self.context = self.alpha @ self.V   # (T, d_model)

        # Project context to get the final output of the self-attention layer
        output = self.context @ self.W_o     # (T, d_model)
        self.output = output
        return output, self.alpha

    def backward(self, doutput):
        """
        Simplified backward pass for demonstration.
        doutput: (T, d_model) gradient wrt attention output.
        Returns:
            dH: (T, d_model) gradient wrt input H.
        """
        T, d_model = self.H.shape
        dk = np.sqrt(self.d_model)

        # Gradients for W_o and context
        self.dW_o = self.context.T @ doutput          # (d_model, d_model)
        dcontext = doutput @ self.W_o.T               # (T, d_model)

        # Gradients for V
        dV = self.alpha.T @ dcontext                  # (T, d_model)

        # Gradients for attention scores via alpha
        # scores = Q K^T / sqrt(dk)
        # context = alpha V
        dscores = (dcontext @ self.V.T) / dk          # (T, T)

        # Row-wise softmax backward approximation:
        # dalpha = dscores * alpha * (1 - alpha)
        dalpha = dscores * self.alpha * (1.0 - self.alpha)   # (T, T)

        # Gradients for Q and K
        dQ = dalpha @ self.K               # (T, d_model)
        dK = dalpha.T @ self.Q             # (T, d_model)

        # Parameter gradients for W_q, W_k, W_v
        self.dW_q = self.H.T @ dQ          # (d_model, d_model)
        self.dW_k = self.H.T @ dK          # (d_model, d_model)
        self.dW_v = self.H.T @ dV          # (d_model, d_model)

        # Gradient w.r.t input H (sum of contributions from Q, K, V)
        dH = dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T  # (T, d_model)
        return dH


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

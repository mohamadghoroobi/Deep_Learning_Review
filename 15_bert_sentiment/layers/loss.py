import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softmax_cross_entropy(logits, targets, pad_token=0):
    B, T, V = logits.shape

    # Stable softmax
    shifted_logits = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Get probability of target tokens
    target_probs = np.take_along_axis(probs, targets[:, :, None], axis=-1).squeeze(-1)

    # Cross entropy loss
    log_probs = -np.log(target_probs + 1e-9)

    # Mask padding tokens
    mask = (targets != pad_token).astype(float)
    loss = (log_probs * mask).sum() / (mask.sum() + 1e-9)

    # Gradient
    d_logits = probs.copy()
    batch_idx = np.arange(B)[:, None]
    time_idx = np.arange(T)[None, :]
    d_logits[batch_idx, time_idx, targets] -= 1
    d_logits = d_logits * mask[:, :, None]
    d_logits /= (B * T)

    return loss, d_logits


def binary_cross_entropy(logits, labels):
    """Binary classification loss for sentiment analysis"""
    # Sigmoid
    probs = 1 / (1 + np.exp(-logits))

    # Binary cross entropy
    loss = -np.mean(labels * np.log(probs + 1e-9) + (1 - labels) * np.log(1 - probs + 1e-9))

    # Gradient
    d_logits = probs - labels

    return loss, d_logits
import numpy as np


def softmax(x):
    """Stable softmax"""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits, targets, pad_token=0):
    """
    Cross entropy loss with padding mask

    Args:
        logits: (B, T, V)
        targets: (B, T)
        pad_token: Padding token ID

    Returns:
        loss, d_logits
    """
    B, T, V = logits.shape

    # Softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)

    # Loss
    target_probs = np.take_along_axis(probs, targets[:, :, None], axis=-1).squeeze(-1)
    log_probs = -np.log(target_probs + 1e-9)
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
    """Binary cross entropy loss"""
    probs = 1 / (1 + np.exp(-logits))
    loss = -np.mean(labels * np.log(probs + 1e-9) + (1 - labels) * np.log(1 - probs + 1e-9))
    d_logits = probs - labels
    return loss, d_logits
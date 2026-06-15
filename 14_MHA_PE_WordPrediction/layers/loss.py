# layers/loss.py
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softmax_cross_entropy(logits, targets, pad_token=0):
    """
    Compute softmax cross entropy loss and gradient.

    Args:
        logits: (B, T, V) - raw scores
        targets: (B, T) - target token indices
        pad_token: int - token to ignore in loss

    Returns:
        loss: float
        d_logits: (B, T, V) - gradient w.r.t logits
    """
    B, T, V = logits.shape

    # ---------- FORWARD: Compute loss ----------
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

    # ---------- BACKWARD: Compute gradient ----------
    d_logits = probs.copy()

    # Subtract 1 at positions of target tokens
    # FIX: Use broadcastable index arrays
    batch_idx = np.arange(B)[:, None]  # Shape: (B, 1)
    time_idx = np.arange(T)[None, :]  # Shape: (1, T)
    d_logits[batch_idx, time_idx, targets] -= 1

    # Apply mask to zero out padding positions
    d_logits = d_logits * mask[:, :, None]

    # Normalize by total number of positions
    d_logits /= (B * T)

    return loss, d_logits
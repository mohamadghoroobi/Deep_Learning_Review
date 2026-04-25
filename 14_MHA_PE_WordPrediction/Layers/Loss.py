import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def softmax_cross_entropy(logits, targets, pad_token=0):
    B,T,V = logits.shape

    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)

    log_probs = -np.log(np.take_along_axis(probs, targets[:,:,None], axis=-1).squeeze(-1) + 1e-9)

    mask = (targets != pad_token)
    loss = (log_probs * mask).sum() / mask.sum()

    d_logits = probs
    idx = targets[:,:,None]
    d_logits[np.arange(B)[:,None], np.arange(T), idx] -= 1

    d_logits /= B*T

    return loss, d_logits

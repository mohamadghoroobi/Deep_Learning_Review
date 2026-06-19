import numpy as np

def create_padding_mask(seq, pad_token=0):
    """Create mask for padding tokens (BERT uses this)"""
    return (seq == pad_token).astype(np.float32)[:, np.newaxis, np.newaxis, :] * -1e9

def create_causal_mask(T):
    """Causal mask (not used in BERT, but included for reference)"""
    return np.triu(np.ones((T, T)), k=1)
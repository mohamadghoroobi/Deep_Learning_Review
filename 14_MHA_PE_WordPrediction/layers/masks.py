import numpy as np

def create_causal_mask(T):
    mask = np.triu(np.ones((T, T)), k=1)
    return mask

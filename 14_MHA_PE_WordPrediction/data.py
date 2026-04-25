import numpy as np

def build_dataset(vocab_size=5000, num_sequences=20000, seq_len=64):
    return np.random.randint(1, vocab_size, size=(num_sequences, seq_len), dtype=np.int32)

def get_batch(data, batch_size):
    idx = np.random.randint(0, len(data), batch_size)
    return data[idx]

"""
Data generation for Modern GPT training
"""

import numpy as np


def build_vocab():
    """Build a simple vocabulary"""
    words = [
        'PAD', 'BOS', 'EOS', 'UNK',
        # Numbers 0-9
        'zero', 'one', 'two', 'three', 'four',
        'five', 'six', 'seven', 'eight', 'nine',
        # Common words
        'the', 'cat', 'dog', 'bird', 'fish',
        'red', 'blue', 'green', 'yellow', 'black',
        'big', 'small', 'fast', 'slow', 'happy',
        'sad', 'angry', 'tired', 'hungry', 'thirsty'
    ]
    return {word: i for i, word in enumerate(words)}


def create_synthetic_data(vocab, num_samples=10000, seq_len=32):
    """
    Create synthetic data for next-token prediction

    Simple task: Predict next number in sequence
    Example: [1, 2, 3, 4, 5] → predict [2, 3, 4, 5, 6]
    """
    vocab_size = len(vocab)
    data = []

    for _ in range(num_samples):
        # Start with a random number sequence
        start = np.random.randint(0, 8)
        length = np.random.randint(8, seq_len - 2)

        # Generate sequence
        seq = list(range(start, start + length))

        # Convert to token IDs (0-9 are numbers)
        token_ids = [vocab['BOS']] + [i % 10 for i in seq] + [vocab['EOS']]

        # Pad to seq_len
        if len(token_ids) < seq_len:
            token_ids += [vocab['PAD']] * (seq_len - len(token_ids))
        else:
            token_ids = token_ids[:seq_len]

        data.append(token_ids)

    return np.array(data, dtype=np.int32)


def get_batch(data, batch_size):
    """Get a random batch"""
    idx = np.random.randint(0, len(data), batch_size)
    return data[idx]


def prepare_training_batch(batch):
    """Prepare batch for training"""
    # Input: all tokens except last
    x = batch[:, :-1]
    # Target: all tokens except first
    y = batch[:, 1:]
    return x, y


if __name__ == "__main__":
    # Test data generation
    vocab = build_vocab()
    print(f"Vocab size: {len(vocab)}")

    data = create_synthetic_data(vocab, num_samples=100, seq_len=16)
    print(f"Data shape: {data.shape}")

    x, y = prepare_training_batch(data[:2])
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
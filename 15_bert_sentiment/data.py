import numpy as np


def build_vocab():
    """Create a simple word vocabulary with proper indices"""
    words = [
        'PAD',  # 0
        'CLS',  # 1
        'MASK',  # 2
        'UNK',  # 3
        # Positive words (4-22)
        'good', 'great', 'amazing', 'excellent', 'wonderful',
        'fantastic', 'awesome', 'brilliant', 'perfect', 'lovely',
        'beautiful', 'nice', 'happy', 'glad', 'pleased',
        'enjoy', 'love', 'like', 'recommend',
        # Negative words (23-36)
        'bad', 'terrible', 'awful', 'horrible', 'poor',
        'disappointing', 'worst', 'boring', 'dull', 'waste',
        'hate', 'dislike', 'avoid', 'mediocre',
        # Neutral words (37-50)
        'movie', 'film', 'actor', 'plot', 'acting',
        'story', 'scene', 'visual', 'music', 'director',
        'cinema', 'theater', 'screen', 'dialog'
    ]
    return {word: i for i, word in enumerate(words)}


def create_sentiment_data(vocab, num_samples=1000, seq_len=20):
    """Create synthetic sentiment dataset"""

    # Get word lists from vocabulary
    positive_words = ['good', 'great', 'amazing', 'excellent', 'wonderful',
                      'fantastic', 'awesome', 'brilliant', 'perfect', 'lovely',
                      'beautiful', 'nice', 'happy', 'glad', 'pleased',
                      'enjoy', 'love', 'like', 'recommend']

    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor',
                      'disappointing', 'worst', 'boring', 'dull', 'waste',
                      'hate', 'dislike', 'avoid', 'mediocre']

    neutral_words = ['movie', 'film', 'actor', 'plot', 'acting',
                     'story', 'scene', 'visual', 'music', 'director',
                     'cinema', 'theater', 'screen', 'dialog']

    def generate_review(sentiment, length):
        if sentiment == 1:  # Positive
            # Mix positive + neutral
            pool = positive_words + neutral_words * 2  # More neutral words
        else:  # Negative
            pool = negative_words + neutral_words * 2

        # Ensure at least one sentiment word
        if sentiment == 1:
            first_word = np.random.choice(positive_words)
        else:
            first_word = np.random.choice(negative_words)

        # Fill rest
        rest = np.random.choice(pool, length - 1)
        words = [first_word] + list(rest)
        return words

    data = []
    labels = []

    for _ in range(num_samples):
        # Random sentiment
        sentiment = np.random.randint(0, 2)

        # Random length (leave room for CLS)
        length = np.random.randint(3, seq_len - 1)

        # Generate review
        review_words = generate_review(sentiment, length)

        # Convert to token IDs - USE VOCAB LOOKUP
        token_ids = [vocab['CLS']]  # Start with CLS token
        for word in review_words:
            # Only add if word exists in vocab
            if word in vocab:
                token_ids.append(vocab[word])
            else:
                token_ids.append(vocab['UNK'])

        # Pad or truncate to seq_len
        if len(token_ids) < seq_len:
            token_ids += [vocab['PAD']] * (seq_len - len(token_ids))
        else:
            token_ids = token_ids[:seq_len]

        data.append(token_ids)
        labels.append(sentiment)

    return np.array(data, dtype=np.int32), np.array(labels, dtype=np.int32)


def get_batch(data, labels, batch_size):
    """Get random batch"""
    idx = np.random.randint(0, len(data), batch_size)
    return data[idx], labels[idx]


# Test the data generation
if __name__ == "__main__":
    vocab = build_vocab()
    print(f"Vocab size: {len(vocab)}")
    print(f"Indices: {min(vocab.values())} to {max(vocab.values())}")

    data, labels = create_sentiment_data(vocab, num_samples=100, seq_len=10)
    print(f"Data shape: {data.shape}")
    print(f"Max token ID: {data.max()}")
    print(f"Should be < {len(vocab)}: {data.max() < len(vocab)}")

    # Print a sample
    sample_idx = 0
    print(f"\nSample {sample_idx}:")
    print(f"Tokens: {data[sample_idx]}")
    print(f"Label: {labels[sample_idx]}")
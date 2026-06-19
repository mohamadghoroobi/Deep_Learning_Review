import numpy as np
from data import build_vocab, create_sentiment_data, get_batch
from model import BERTEncoder
from layers.loss import softmax_cross_entropy, binary_cross_entropy

# Parameters
vocab_size = 5000
d_model = 64  # Smaller for CPU
num_heads = 2
d_ff = 256
num_layers = 2
max_len = 32
batch_size = 16
epochs = 50
lr = 0.001
seq_len = 20

# Build vocabulary and dataset
vocab = build_vocab()
data, labels = create_sentiment_data(vocab, num_samples=2000, seq_len=seq_len)

# Split train/val
split = int(0.8 * len(data))
train_data, val_data = data[:split], data[split:]
train_labels, val_labels = labels[:split], labels[split:]

print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
print(f"Vocab size: {len(vocab)}")

# Create model
model = BERTEncoder(
    vocab_size=len(vocab),
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_len=max_len,
    dropout=0.1
)

# Training loop
for epoch in range(epochs):
    # Get batch
    x_batch, y_batch = get_batch(train_data, train_labels, batch_size)

    # Forward pass
    logits, _ = model.forward(x_batch)

    # Binary cross entropy loss
    loss, d_logits = binary_cross_entropy(logits, y_batch[:, None])

    # Backward pass
    model.backward(d_logits)

    # Update weights
    model.update(lr)

    # Validation every 10 epochs
    if epoch % 10 == 0:
        # Compute accuracy on validation set
        val_logits, _ = model.forward(val_data[:100])
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_preds == val_labels[:100])
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Val Acc = {val_acc:.2%}")

print("Training complete!")

# Save model (optional)
import pickle
with open('bert_sentiment.pkl', 'wb') as f:
    pickle.dump(model, f)
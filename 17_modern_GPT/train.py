"""
Training script for Modern GPT
"""

import numpy as np
import pickle
from config import ModelConfig, DataConfig
from model import ModernGPT
from data import build_vocab, create_synthetic_data, get_batch, prepare_training_batch
from layers.loss import cross_entropy


def train():
    """Main training loop"""
    print("=" * 60)
    print("MODERN GPT TRAINING")
    print("=" * 60)

    # Load configs
    model_config = ModelConfig()
    data_config = DataConfig()

    # Build vocabulary
    vocab = build_vocab()
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # Create data
    print(f"Creating synthetic data...")
    data = create_synthetic_data(vocab, data_config.num_samples, model_config.max_len)
    print(f"Data shape: {data.shape}")

    # Split train/validation
    split = int(data_config.split_ratio * len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Create model
    print("\nCreating model...")
    model = ModernGPT(
        vocab_size=vocab_size,
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        max_len=model_config.max_len,
        dropout=model_config.dropout
    )

    # Print model info (rough estimate)
    total_params = 0
    for attr in model.__dict__.values():
        if hasattr(attr, 'update'):
            if hasattr(attr, 'E'):
                total_params += attr.E.size
            elif hasattr(attr, 'W'):
                total_params += attr.W.size
    print(f"Total parameters: ~{total_params:,}")

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(model_config.epochs):
        # Get batch
        batch = get_batch(train_data, model_config.batch_size)
        x, y = prepare_training_batch(batch)

        # Forward pass
        logits = model.forward(x)

        # Loss
        loss, d_logits = cross_entropy(logits, y, pad_token=0)

        # Backward pass
        model.backward(d_logits)

        # Update weights
        model.update(model_config.learning_rate)

        # Validation every 10 epochs
        if epoch % 10 == 0:
            val_batch = get_batch(val_data, model_config.batch_size)
            val_x, val_y = prepare_training_batch(val_batch)

            val_logits = model.forward(val_x)
            val_loss, _ = cross_entropy(val_logits, val_y, pad_token=0)

            preds = np.argmax(val_logits, axis=-1)
            mask = val_y != 0
            acc = np.mean((preds == val_y)[mask])

            print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, Acc = {acc:.2%}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with open('modern_gpt_best.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Save final model
    with open('modern_gpt_final.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Final model saved as modern_gpt_final.pkl")

    return model


if __name__ == "__main__":
    model = train()
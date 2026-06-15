import numpy as np
from model import TransformerLM
from data import build_dataset, get_batch
from layers.loss import softmax_cross_entropy, softmax

vocab_size = 5000
seq_len = 64
dataset = build_dataset(vocab_size, num_sequences=30000, seq_len=seq_len)

batch_size = 32
epochs = 20
lr = 1e-3
PAD = 0


model = TransformerLM(vocab_size, d_model=128, num_heads=4, d_ff=512, num_Layers=3)


def generate(model, start_tokens, max_new=40):
    x = np.array([start_tokens], dtype=np.int32)
    for _ in range(max_new):
        logits = model.forward(x)
        last = logits[0,-1]
        probs = softmax(last)
        next_token = np.random.choice(vocab_size, p=probs)
        x = np.concatenate([x, [[next_token]]], axis=1)
    return x[0]


# -------- Training loop --------
for ep in range(epochs):
    batch = get_batch(dataset, batch_size)

    inp = batch[:, :-1]
    tgt = batch[:, 1:]

    logits = model.forward(inp)
    loss, d_logits = softmax_cross_entropy(logits, tgt, pad_token=PAD)

    model.backward(d_logits)
    model.update(lr)

    print(f"Epoch {ep+1}/{epochs}   Loss = {loss:.4f}")

# Test-generation
print("\nGenerated sequence:\n", generate(model, [10, 20, 30], max_new=30))

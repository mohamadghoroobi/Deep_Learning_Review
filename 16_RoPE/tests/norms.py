import numpy as np


def compare_norm(d_model=64, seq_len=20):
    # 1. Sinusoidal PE (Additive)
    class SinusoidalPE:
        def __init__(self, d_model):
            self.d_model = d_model
            self.pe = np.zeros((100, d_model))
            pos = np.arange(100).reshape(-1, 1)
            div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            self.pe[:, 0::2] = np.sin(pos * div)
            self.pe[:, 1::2] = np.cos(pos * div)

        def forward(self, x, pos):
            return x + self.pe[pos:pos + len(x)]

    # 2. RoPE (Rotary)
    class RoPE:
        def __init__(self, d_model):
            self.d_model = d_model
            self.freqs = 10000 ** (-2.0 * np.arange(d_model // 2) / d_model)

        def forward(self, x, pos):
            # Apply rotation
            angles = pos.reshape(-1, 1) * self.freqs  # (T, D/2)
            cos = np.cos(angles)
            sin = np.sin(angles)

            # Split and rotate
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]

            x1_new = x1 * cos - x2 * sin
            x2_new = x1 * sin + x2 * cos

            # Interleave
            rotated = np.stack([x1_new, x2_new], axis=-1)
            return rotated.reshape(*x.shape)

    # Create random embeddings
    embeddings = np.random.randn(seq_len, d_model)

    # Apply both PEs
    sine_pe = SinusoidalPE(d_model)
    rope = RoPE(d_model)

    sinusoidal_out = sine_pe.forward(embeddings, pos=0)
    rope_out = rope.forward(embeddings, pos=np.arange(seq_len))

    # Compute norms
    original_norms = np.linalg.norm(embeddings, axis=1)
    sinusoidal_norms = np.linalg.norm(sinusoidal_out, axis=1)
    rope_norms = np.linalg.norm(rope_out, axis=1)

    print("Norm Comparison:")
    print("-" * 50)
    print(f"Original norm:   {original_norms[0]:.4f} ... {original_norms[-1]:.4f}")
    print(f"Sinusoidal PE:   {sinusoidal_norms[0]:.4f} ... {sinusoidal_norms[-1]:.4f}")
    print(f"RoPE:            {rope_norms[0]:.4f} ... {rope_norms[-1]:.4f}")

    print("\nNorm change (Sinusoidal):",
          f"{np.std(sinusoidal_norms - original_norms):.4f}")
    print("Norm change (RoPE):",
          f"{np.std(rope_norms - original_norms):.4f}")

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(original_norms, 'b-', label='Original')
    plt.plot(sinusoidal_norms, 'r--', label='Sinusoidal PE')
    plt.plot(rope_norms, 'g--', label='RoPE')
    plt.xlabel('Position')
    plt.ylabel('Norm')
    plt.legend()
    plt.title('Norm Preservation: Sinusoidal vs RoPE')
    plt.show()


# Run comparison
compare_norm()
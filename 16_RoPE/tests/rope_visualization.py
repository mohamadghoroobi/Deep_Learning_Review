import numpy as np
import matplotlib.pyplot as plt
from rotatory_positional_embedding import RotaryPositionalEmbedding


def visualize_rope_with_synthetic_meaningful_vectors():
    """
    Visualize RoPE with vectors that simulate meaningful relationships
    """
    d_model = 64
    rope = RotaryPositionalEmbedding(d_model)

    # Create q that represents "a meaningful token"
    q = np.random.randn(d_model)
    q = q / np.linalg.norm(q) * 5

    # Create k that is similar to q (represents a related token)
    # This simulates "cat" and "sat" - related tokens
    k = q + 0.5 * np.random.randn(d_model)
    k = k / np.linalg.norm(k) * 5

    # Test all distances
    max_dist = 30
    dots = []

    for dist in range(max_dist):
        dot_values = []
        for start in range(0, 60 - dist, 3):
            m = start
            n = start + dist
            q_m = rope.forward(q[None, None, :], start_pos=m)[0, 0]
            k_n = rope.forward(k[None, None, :], start_pos=n)[0, 0]
            dot = np.dot(q_m, k_n)
            dot_values.append(dot)
        dots.append(np.mean(dot_values))

    # Also compute the theoretical curve
    base_dot = np.dot(q, k)  # Base dot product
    theoretical = [base_dot * np.cos(dist) for dist in range(max_dist)]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dots, 'b-', linewidth=2, label='Actual (with positional variation)')
    plt.plot(theoretical, 'r--', linewidth=1.5, label='Theoretical: base * cos(dist)')
    plt.xlabel('Distance between tokens')
    plt.ylabel('Dot Product (Attention Score)')
    plt.title('RoPE: Attention vs Distance (with Meaningful Vectors)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend()

    # Add annotations
    plt.annotate('Close tokens: Higher attention',
                 xy=(1, dots[1]), xytext=(5, max(dots) * 0.6),
                 arrowprops=dict(arrowstyle='->', color='green'))

    plt.annotate('Far tokens: Lower attention',
                 xy=(20, dots[20]), xytext=(15, min(dots) * 0.8),
                 arrowprops=dict(arrowstyle='->', color='red'))

    # Show which distances are "important" for language
    important_distances = [1, 2, 3, 4, 5]  # Nearby tokens
    for d in important_distances:
        plt.axvline(x=d, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.show()


# Run
visualize_rope_with_synthetic_meaningful_vectors()
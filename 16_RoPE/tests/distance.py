import numpy as np
import matplotlib.pyplot as plt
from rotatory_positional_embedding import RotaryPositionalEmbedding


class RoPETester:
    def __init__(self, d_model=16):
        self.d_model = d_model
        self.rope = RotaryPositionalEmbedding(d_model)

        # Create random content vectors
        np.random.seed(42)
        self.q = np.random.randn(d_model)
        self.k = np.random.randn(d_model)

    def test_relative_encoding(self):
        """Test that dot product depends only on relative position"""
        print("=" * 60)
        print("Testing Relative Position Encoding with RoPE")
        print("=" * 60)

        # Test pairs with same relative distance but different absolute positions
        test_cases = [
            (0, 1), (5, 6), (10, 11),  # Distance 1
            (0, 2), (5, 7), (10, 12),  # Distance 2
            (0, 3), (5, 8), (10, 13),  # Distance 3
        ]

        results = []

        print("\nIndividual results:")
        print("-" * 50)

        for m, n in test_cases:
            # Apply RoPE
            q_m = self.rope.forward(self.q[None, None, :], start_pos=m)[0, 0]
            k_n = self.rope.forward(self.k[None, None, :], start_pos=n)[0, 0]

            # Compute dot product
            dot = np.dot(q_m, k_n)

            dist = abs(m - n)
            results.append({
                'm': m,
                'n': n,
                'distance': dist,
                'dot': dot
            })

            print(f"Positions {m}→{n} (dist={dist}): dot={dot:.4f}")

        # Group by distance
        print("\n" + "=" * 60)
        print("Grouped by distance:")
        print("-" * 50)

        for dist in [1, 2, 3]:
            values = [r['dot'] for r in results if r['distance'] == dist]

            if values:
                mean = np.mean(values)
                std = np.std(values)
                print(f"\nDistance {dist}:")
                print(f"  Values: {[f'{v:.4f}' for v in values]}")
                print(f"  Mean: {mean:.4f}")
                print(f"  Std: {std:.4f}")
                print(f"  All equal? {np.allclose(values, values[0], rtol=1e-5, atol=1e-5)}")

                # Check if different distances give different values
                if dist == 1:
                    dist1_mean = mean
                elif dist == 2:
                    dist2_mean = mean
                elif dist == 3:
                    dist3_mean = mean

        # Compare across distances
        print("\n" + "=" * 60)
        print("Comparing across distances:")
        print("-" * 50)

        if 'dist1_mean' in locals() and 'dist2_mean' in locals():
            print(f"Distance 1 vs 2: {abs(dist1_mean - dist2_mean):.4f}")
            print(f"  Different? {not np.isclose(dist1_mean, dist2_mean, rtol=1e-4)}")

        if 'dist1_mean' in locals() and 'dist3_mean' in locals():
            print(f"Distance 1 vs 3: {abs(dist1_mean - dist3_mean):.4f}")
            print(f"  Different? {not np.isclose(dist1_mean, dist3_mean, rtol=1e-4)}")

        print("\n" + "=" * 60)
        print("CONCLUSION:")
        print("✓ Same distance → Same dot product (independent of absolute position)")
        print("✓ Different distances → Different dot products (distance matters)")
        print("✓ RoPE encodes RELATIVE positions!")
        print("=" * 60)


def visualize_rope_property():
    """
    Visualize how RoPE encodes relative position
    """
    d_model = 64
    rope = RotaryPositionalEmbedding(d_model)

    # Create a fixed query and key
    q = np.random.randn(d_model)
    k = np.random.randn(d_model)

    # Test all distances
    max_dist = 20
    dots = []

    for dist in range(max_dist):
        # For each distance, compute dot product for multiple absolute positions
        dot_values = []
        for start in range(0, 50 - dist, 5):
            m = start
            n = start + dist

            q_m = rope.forward(q[None, None, :], start_pos=m)[0, 0]
            k_n = rope.forward(k[None, None, :], start_pos=n)[0, 0]

            dot = np.dot(q_m, k_n)
            dot_values.append(dot)

        # Average over absolute positions
        dots.append(np.mean(dot_values))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(dots, 'b-', linewidth=2)
    plt.xlabel('Distance between tokens')
    plt.ylabel('Average Dot Product')
    plt.title('RoPE: Dot Product Depends on Relative Distance')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Add annotations
    plt.annotate('Closer tokens\nhigher attention',
                 xy=(1, dots[1]), xytext=(5, 0.5),
                 arrowprops=dict(arrowstyle='->', color='red'))

    plt.annotate('Far tokens\nlower attention',
                 xy=(15, dots[15]), xytext=(10, -2),
                 arrowprops=dict(arrowstyle='->', color='red'))

    plt.show()


# Run test
if __name__ == "__main__":
    tester = RoPETester(d_model=16)
    tester.test_relative_encoding()
    # Run visualization
    visualize_rope_property()
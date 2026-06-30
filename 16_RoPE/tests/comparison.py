import numpy as np
import matplotlib.pyplot as plt

# Sinusoidal PE (Absolute)
def sin_pe_dot(m, n, d_model=64):
    """Compute dot product of sinusoidal PE at positions m and n"""
    pe_m = np.zeros(d_model)
    pe_n = np.zeros(d_model)
    for i in range(0, d_model, 2):
        pe_m[i] = np.sin(m / 10000**(2*i/d_model))
        pe_m[i+1] = np.cos(m / 10000**(2*i/d_model))
        pe_n[i] = np.sin(n / 10000**(2*i/d_model))
        pe_n[i+1] = np.cos(n / 10000**(2*i/d_model))
    return np.dot(pe_m, pe_n)

# RoPE
def rope_effect(dist, d_model=64):
    """RoPE modulation depends only on distance"""
    return np.cos(dist)  # Simplified

# Compare
positions = np.arange(0, 20)
distances = np.arange(0, 20)

# For Sinusoidal PE: depends on m and n separately
sin_pe_values = []
for m in range(20):
    values = []
    for n in range(20):
        values.append(sin_pe_dot(m, n))
    sin_pe_values.append(values)

# For RoPE: depends only on distance
rope_values = [rope_effect(d) for d in distances]

# Plot
plt.figure(figsize=(14, 5))

# Sinusoidal PE heatmap
plt.subplot(1, 2, 1)
plt.imshow(sin_pe_values, cmap='coolwarm', aspect='auto')
plt.xlabel('Position n')
plt.ylabel('Position m')
plt.title('Sinusoidal PE: Depends on ABSOLUTE positions\nSame diagonal distance = different values!')
plt.colorbar()

# RoPE
plt.subplot(1, 2, 2)
plt.plot(distances, rope_values, 'r-', linewidth=2)
plt.xlabel('Distance (m - n)')
plt.ylabel('Attention modulation')
plt.title('RoPE: Depends ONLY on RELATIVE distance\nAll pairs with same distance = same value!')
plt.grid(True)

plt.tight_layout()
plt.show()
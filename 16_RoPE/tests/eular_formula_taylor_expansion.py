import numpy as np
import matplotlib.pyplot as plt

# Plot e^(iθ) for θ from 0 to 2π
theta = np.linspace(0, 2*np.pi, 100)
real = np.cos(theta)
imag = np.sin(theta)

plt.figure(figsize=(8, 8))
plt.plot(real, imag, 'b-', linewidth=2)
plt.plot(1, 0, 'ro', label='θ=0')
plt.plot(0, 1, 'go', label='θ=π/2')
plt.plot(-1, 0, 'yo', label='θ=π')
plt.plot(0, -1, 'co', label='θ=3π/2')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Real (cos θ)')
plt.ylabel('Imaginary (sin θ)')
plt.title('e^(iθ) = cos(θ) + i*sin(θ)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def dtft(x, w):
    """
    Compute the Discrete-Time Fourier Transform (DTFT) of a signal x at frequencies w.
    
    Parameters:
    - x: The discrete time-domain signal
    - w: The array of frequencies in radians
    
    Returns:
    - X: The DTFT of the signal at frequencies w
    """
    N = len(x)
    X = np.zeros_like(w, dtype=complex)
    for k, freq in enumerate(w):
        X[k] = np.sum(x * np.exp(-1j * freq * np.arange(N)))
    return X

# Define the signal
n = np.arange(0, 500)
x = np.sin(2 * np.pi * 200 / 8000 * n)

# Define the frequency range for DTFT
w = np.arange(-np.pi, np.pi, 0.0001 * np.pi)

# Compute the DTFT
X = dtft(x, w)

# Compute magnitude and phase
magnitude = np.abs(X)
phase = np.angle(X)

# Plot magnitude
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(w, magnitude)
plt.title('Magnitude of DTFT')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot phase
plt.subplot(2, 1, 2)
plt.plot(w, phase)
plt.title('Phase of DTFT')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.grid(True)

plt.tight_layout()
plt.show()


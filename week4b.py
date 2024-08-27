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
x = np.array([1, 2, 3, 4])

# Define the time shift amount
n0 = 2  # Amount of time shift

# Define the frequency range for DTFT
w = np.arange(-np.pi, np.pi, 0.0001 * np.pi)

# Compute the DTFT of the original signal
X_original = dtft(x, w)

# Compute the time-shifted signal
x_shifted = np.roll(x, shift=n0)  # Time shift the signal
X_shifted = dtft(x_shifted, w)

# Compute the theoretical DTFT of the shifted signal
X_shifted_theoretical = np.exp(-1j * w * n0) * X_original

# Plot DTFTs and verify time shift
plt.figure(figsize=(12, 6))

# Plot magnitude of DTFT of original signal
plt.subplot(2, 2, 1)
plt.plot(w, np.abs(X_original))
plt.title('Magnitude of DTFT of Original Signal')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot phase of DTFT of original signal
plt.subplot(2, 2, 2)
plt.plot(w, np.angle(X_original))
plt.title('Phase of DTFT of Original Signal')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.grid(True)

# Plot magnitude of DTFT of time-shifted signal
plt.subplot(2, 2, 3)
plt.plot(w, np.abs(X_shifted))
plt.title('Magnitude of DTFT of Time-Shifted Signal')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude')
plt.grid(True)

# Plot phase of DTFT of time-shifted signal
plt.subplot(2, 2, 4)
plt.plot(w, np.angle(X_shifted))
plt.title('Phase of DTFT of Time-Shifted Signal')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Verify the time shift property
plt.figure(figsize=(12, 6))

# Plot magnitude of theoretical and computed DTFT of shifted signal
plt.subplot(2, 2, 1)
plt.plot(w, np.abs(X_shifted), label='Computed')
plt.plot(w, np.abs(X_shifted_theoretical), '--', label='Theoretical', color='orange')
plt.title('Magnitude Comparison')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

# Plot phase of theoretical and computed DTFT of shifted signal
plt.subplot(2, 2, 2)
plt.plot(w, np.angle(X_shifted), label='Computed')
plt.plot(w, np.angle(X_shifted_theoretical), '--', label='Theoretical', color='orange')
plt.title('Phase Comparison')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


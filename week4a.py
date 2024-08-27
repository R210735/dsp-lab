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

# Define the signals
X1 = np.array([1, 2, 3, 4])
X2 = np.array([4, 3, 2, 1])

# Define the frequency range for DTFT
w = np.arange(-np.pi, np.pi, 0.0001 * np.pi)

# Compute the DTFT of the signals
X1_dtft = dtft(X1, w)
X2_dtft = dtft(X2, w)

# Compute the DTFT of the sum of X1 and X2
X_sum = X1 + X2
X_sum_dtft = dtft(X_sum, w)

# Compute the DTFT of the sum of the individual DTFTs
X1_X2_sum_dtft = X1_dtft + X2_dtft

# Plot DTFTs and verify linearity
plt.figure(figsize=(12, 6))

# Plot DTFT of X1
plt.subplot(3, 2, 1)
plt.plot(w, np.abs(X1_dtft))
plt.title('Magnitude of DTFT of X1')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(w, np.angle(X1_dtft))
plt.title('Phase of DTFT of X1')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.grid(True)

# Plot DTFT of X2
plt.subplot(3, 2, 3)
plt.plot(w, np.abs(X2_dtft))
plt.title('Magnitude of DTFT of X2')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(w, np.angle(X2_dtft))
plt.title('Phase of DTFT of X2')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.grid(True)

# Plot DTFT of X1 + X2
plt.subplot(3, 2, 5)
plt.plot(w, np.abs(X_sum_dtft))
plt.title('Magnitude of DTFT of X1 + X2')
plt.xlabel('Frequency (radians)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(w, np.angle(X_sum_dtft))
plt.title('Phase of DTFT of X1 + X2')
plt.xlabel('Frequency (radians)')
plt.ylabel('Phase (radians)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Verify linearity by comparing DTFT of the sum with the sum of DTFTs
assert np.allclose(np.abs(X_sum_dtft), np.abs(X1_X2_sum_dtft)), "Magnitude does not match"
assert np.allclose(np.angle(X_sum_dtft), np.angle(X1_X2_sum_dtft)), "Phase does not match"

print("Linearity of DTFT verified successfully!")


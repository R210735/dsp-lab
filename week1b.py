import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# Parameters for the original signal
fs_original = 8000        # Original sampling frequency (Hz)
f = 200                   # Frequency of the sinusoid (Hz)
duration = 0.5            # Duration (seconds)
A = 1                     # Amplitude of the sinusoid

# Time vector for the original signal
t_original = np.arange(0, duration, 1/fs_original)  # Time vector for the original signal

# Generate the original sinusoidal signal
x_original = A * np.sin(2 * np.pi * f * t_original)

# Parameters for the new signal
fs_new = 1000             # New sampling frequency (Hz)

# Resample the signal
num_samples_new = int(duration * fs_new)
x_resampled = resample(x_original, num_samples_new)
t_resampled = np.arange(0, duration, 1/fs_new)  # Time vector for the resampled signal

# Plot the original and resampled signals
plt.figure(figsize=(12, 6))

# Plot original signal
plt.subplot(2, 1, 1)
plt.plot(t_original, x_original, label='Original Signal (8000 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original 200 Hz Sinusoid Sampled at 8000 Hz')
plt.grid(True)
plt.legend()

# Plot resampled signal
plt.subplot(2, 1, 2)
plt.plot(t_resampled, x_resampled, label='Resampled Signal (1000 Hz)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Resampled 200 Hz Sinusoid at 1000 Hz')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


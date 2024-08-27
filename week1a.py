import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 8000        # Sampling frequency (Hz)
f = 200          # Frequency of the sinusoid (Hz)
duration = 0.5   # Duration (seconds)
A = 1            # Amplitude of the sinusoid

# Time vector
t = np.arange(0, duration, 1/fs)  # Create time vector from 0 to duration with step 1/fs

# Generate the sinusoidal signal
x = A * np.sin(2 * np.pi * f * t)

# Plot the sinusoidal signal
plt.figure()
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('200 Hz Sinusoid Sampled at 8000 Hz')
plt.grid(True)
plt.show()


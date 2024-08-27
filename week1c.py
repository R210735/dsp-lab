import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# Parameters
fs_original = 8000        # Original sampling frequency (Hz)
f = 200                   # Frequency of the sinusoid (Hz)
duration = 0.5            # Duration (seconds)
A = 1                     # Amplitude of the sinusoid
fs_new = 1000             # New sampling frequency (Hz)
L = 8                     # Number of quantization levels

# Generate the original sinusoidal signal
t_original = np.arange(0, duration, 1/fs_original)  # Time vector for the original signal
x_original = A * np.sin(2 * np.pi * f * t_original)

# Resample the signal
num_samples_new = int(duration * fs_new)
x_resampled = resample(x_original, num_samples_new)
t_resampled = np.arange(0, duration, 1/fs_new)  # Time vector for the resampled signal

# Quantize the signal
def quantize(signal, levels):
    # Normalize the signal to the range [0, 1]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalized_signal = (signal - signal_min) / (signal_max - signal_min)
    
    # Quantize the normalized signal
    quantized_signal = np.floor(normalized_signal * (levels - 1))
    return quantized_signal

quantized_signal = quantize(x_resampled, L)

# Convert quantized values to binary form
def quantized_to_binary(quantized_signal):
    binary_values = [format(int(value), f'0{int(np.log2(L))}b') for value in quantized_signal]
    return ''.join(binary_values)

binary_signal = quantized_to_binary(quantized_signal)

# Save binary data to a file
with open('quantized_signal.bin', 'wb') as file:
    # Convert binary string to bytes and write to file
    file.write(int(binary_signal, 2).to_bytes((len(binary_signal) + 7) // 8, byteorder='big'))

# Plot the resampled and quantized signals
plt.figure(figsize=(12, 6))

# Plot resampled signal
plt.subplot(2, 1, 1)
plt.plot(t_resampled, x_resampled, label='Resampled Signal (1000 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Resampled 200 Hz Sinusoid at 1000 Hz')
plt.grid(True)
plt.legend()

# Plot quantized signal
plt.subplot(2, 1, 2)
plt.step(t_resampled, quantized_signal, label='Quantized Signal (8 Levels)', where='post', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Quantized Level')
plt.title('Quantized Signal with 8 Levels')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def dft(x, N):
    X = np.fft.fft(x, N)
    return X

def plot_dft(X, fs):
    w = np.fft.fftfreq(len(X), d=1/fs)
    magnitude = np.abs(X)
    phase = np.angle(X)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(w, magnitude)
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(w, phase)
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    peak_value = np.max(magnitude)
    print(f'Peak value in magnitude spectrum: {peak_value}')

# Define the signal
x = np.array([1, 2, 3, 4])
N = len(x)

# Compute and plot DFT of the signal
X = dft(x, N)
plot_dft(X, fs=1)

# Perform DFT for an audio file
input_filename = 'input_audio.wav'
fs, audio_data = wavfile.read(input_filename)

# For large audio files, use only a segment for DFT
segment = audio_data[:2048]  # Use first 2048 samples or adjust as needed
X_audio = dft(segment, len(segment))

# Plot the magnitude and phase of the audio signal
plot_dft(X_audio, fs)


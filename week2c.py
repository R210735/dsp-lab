import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

def load_audio(filename):
    fs, data = wavfile.read(filename)
    return fs, data

def plot_waveform(data, fs, title, subplot):
    t = np.arange(len(data)) / fs
    plt.subplot(subplot)
    plt.plot(t, data, label=title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.legend()

def play_audio(data, fs):
    sd.play(data, fs)
    sd.wait()  # Wait until the sound has finished playing

# File paths
original_filename = 'input_audio.wav'
upsampled_filename = 'upsampled_audio.wav'
downsampled_filename = 'downsampled_audio.wav'

# Load audio files
original_fs, original_audio = load_audio(original_filename)
upsampled_fs, upsampled_audio = load_audio(upsampled_filename)
downsampled_fs, downsampled_audio = load_audio(downsampled_filename)

# Plot waveforms
plt.figure(figsize=(15, 10))

plot_waveform(original_audio, original_fs, 'Original Audio', 311)
plot_waveform(upsampled_audio, upsampled_fs, 'Upsampled Audio', 312)
plot_waveform(downsampled_audio, downsampled_fs, 'Downsampled Audio', 313)

plt.tight_layout()
plt.show()

# Play audio files
print("Playing original audio...")
play_audio(original_audio, original_fs)

print("Playing upsampled audio...")
play_audio(upsampled_audio, upsampled_fs)

print("Playing downsampled audio...")
play_audio(downsampled_audio, downsampled_fs)


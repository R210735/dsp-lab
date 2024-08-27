import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

def downsampling(audio_data, original_fs, new_fs):
    """
    Downsample the audio signal from original_fs to new_fs.
    
    Parameters:
    - audio_data: numpy array of the audio signal
    - original_fs: original sampling frequency
    - new_fs: new sampling frequency
    
    Returns:
    - downsampled_audio: numpy array of the downsampled audio signal
    """
    # Calculate the number of samples in the downsampled signal
    num_samples_new = int(len(audio_data) * (new_fs / original_fs))
    
    # Perform downsampling
    downsampled_audio = resample(audio_data, num_samples_new)
    
    return downsampled_audio

# Load the original audio file
input_filename = 'input_audio.wav'
output_filename = 'downsampled_audio.wav'
original_fs, audio_data = wavfile.read(input_filename)

# Define new sampling frequency
new_fs = 1000  # Example new sampling frequency (Hz)

# Downsample the audio data
downsampled_audio = downsampling(audio_data, original_fs, new_fs)

# Save the downsampled audio to a new file
# Ensure to clip or normalize the audio data to avoid overflow
downsampled_audio = np.clip(downsampled_audio, -32768, 32767)  # Clip to 16-bit range
wavfile.write(output_filename, new_fs, downsampled_audio.astype(np.int16))

print(f"Downsampled audio saved to {output_filename}")


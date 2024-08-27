import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

def upsampling(audio_data, original_fs, new_fs):
    """
    Upsample the audio signal from original_fs to new_fs.
    
    Parameters:
    - audio_data: numpy array of the audio signal
    - original_fs: original sampling frequency
    - new_fs: new sampling frequency
    
    Returns:
    - upsampled_audio: numpy array of the upsampled audio signal
    """
    # Calculate the number of samples in the upsampled signal
    num_samples_new = int(len(audio_data) * (new_fs / original_fs))
    
    # Perform upsampling
    upsampled_audio = resample(audio_data, num_samples_new)
    
    return upsampled_audio

# Load the original audio file
input_filename = '/home/rgukt/Downloads/Sports.wav'
output_filename = 'upsampled_audio.wav'
original_fs, audio_data = wavfile.read(input_filename)

# Define new sampling frequency
new_fs = 16000  # Example new sampling frequency (Hz)

# Upsample the audio data
upsampled_audio = upsampling(audio_data, original_fs, new_fs)

# Save the upsampled audio to a new file
wavfile.write(output_filename, new_fs, upsampled_audio.astype(np.int16))

print(f"Upsampled audio saved to {output_filename}")


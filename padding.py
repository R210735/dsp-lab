import numpy as np
import matplotlib.pyplot as plt
sequence = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
def compute_dft(sequence):
    n = len(sequence)
    dft_result = np.zeros(n, dtype=complex)
    for k in range(n):
        for t in range(n):
            angle = -2j * np.pi * k * t / n
            dft_result[k] += sequence[t] * np.exp(angle)
    return dft_result
padding_values = [2, 4, 8]
plt.figure(figsize=(12, 9))
for i, pad in enumerate(padding_values):
    
    n_padded = pad * len(sequence)  # Zero padding by factor of pad
    sequence_padded = np.pad(sequence, (0, n_padded - len(sequence)), 'constant')
    dft_sequence_padded = compute_dft(sequence_padded)

    
    frequencies_padded = np.arange(len(sequence_padded)) / len(sequence_padded)

    
    plt.subplot(len(padding_values), 1, i + 1)
    plt.stem(frequencies_padded, np.abs(dft_sequence_padded), use_line_collection=True)
    plt.title(f"DFT of Sequence with Zero Padding (Factor = {pad})")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude")
    plt.grid()
plt.tight_layout()
plt.show()

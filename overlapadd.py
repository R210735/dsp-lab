import numpy as np
import matplotlib.pyplot as plt

def overlap_add(x, h, segment_length):
    # Calculate lengths
    L = segment_length
    M = len(h)
    # Ensure segment length is greater than the filter length
    assert L > M, "Segment length must be greater than filter length"
    
    # Pad x with zeros to make sure it fits
    padded_x = np.pad(x, (0, L - len(x) % L), 'constant')
    
    # Number of segments
    num_segments = len(padded_x) // L
    
    # Initialize result
    y = np.zeros(len(padded_x) + M - 1)
    
    # Perform the overlap-add
    for i in range(num_segments):
        start = i * L
        end = start + L
        x_segment = padded_x[start:end]
        
        # Convolve the segment with the filter
        y_segment = np.convolve(x_segment, h, mode='full')
        
        # Add the segment to the result
        y[start:start + len(y_segment)] += y_segment
    
    return y

# Define the sequences
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
h = np.array([1, 2, 1])

# Segment length (must be greater than length of filter)
segment_length = 5

# Perform overlap-add
y_overlap_add = overlap_add(x, h, segment_length)

# Linear convolution (for comparison)
y_linear = np.convolve(x, h, mode='full')

# Plotting the results
plt.figure(figsize=(12, 6))

# Original Signal
plt.subplot(3, 1, 1)
plt.stem(x, use_line_collection=True)
plt.title("Original Signal x(n)")
plt.xlabel("n")
plt.ylabel("x(n)")

# Filter
plt.subplot(3, 1, 2)
plt.stem(h, use_line_collection=True)
plt.title("Filter h(n)")
plt.xlabel("n")
plt.ylabel("h(n)")

# Overlap-Add Convolution
plt.subplot(3, 1, 3)
plt.stem(y_overlap_add, use_line_collection=True, label='Overlap-Add')
plt.stem(y_linear, linefmt='r-', markerfmt='ro', label='Linear Convolution', basefmt='r-')
plt.title("Convolution Results")
plt.xlabel("n")
plt.ylabel("y(n)")
plt.legend()

plt.tight_layout()
plt.show()


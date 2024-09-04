import numpy as np
import matplotlib.pyplot as plt

# Function to compute linear convolution manually
def linear_convolution(x, h):
    M = len(x)
    N = len(h)
    y = np.zeros(M + N - 1)
    
    for n in range(len(y)):
        for k in range(M):
            if 0 <= n - k < N:
                y[n] += x[k] * h[n - k]
    
    return y

# Function to compute circular convolution manually
def circular_convolution(x, h):
    M = len(x)
    N = len(h)
    L = M + N - 1
    
    # Zero-padding
    x_padded = np.concatenate([x, np.zeros(N - 1)])
    h_padded = np.concatenate([h, np.zeros(M - 1)])
    
    # Circular convolution using zero-padding
    y = np.zeros(L)
    for n in range(L):
        for k in range(M):
            y[n] += x_padded[k] * h_padded[(n - k) % L]
    
    return y

# Define the sequences
x = np.array([1, 2, 3])
h = np.array([4, 5])

# Compute linear convolution
y_linear = linear_convolution(x, h)

# Compute circular convolution
y_circular = circular_convolution(x, h)

# Plotting the results
plt.figure(figsize=(12, 6))

# Linear Convolution
plt.subplot(2, 2, 1)
plt.stem(np.arange(len(y_linear)), y_linear, use_line_collection=True)
plt.title("Linear Convolution")
plt.xlabel("n")
plt.ylabel("y(n)")

# Circular Convolution
plt.subplot(2, 2, 2)
plt.stem(np.arange(len(y_circular)), y_circular, use_line_collection=True)
plt.title("Circular Convolution")
plt.xlabel("n")
plt.ylabel("y(n)")

# Check if they are equal
plt.subplot(2, 1, 2)
plt.plot(np.abs(y_linear - y_circular), 'o')
plt.title("Difference between Linear and Circular Convolution")
plt.xlabel("Index")
plt.ylabel("Difference")

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def convolve(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    for n in range(len(y)):
        for m in range(M):
            if 0 <= n - m < N:
                y[n] += x[n - m] * h[m]
    return y

x = np.array([1, 2, 3, 4])
h = np.array([0.2, 0.5, 0.3])

y = convolve(x, h)

plt.figure(figsize=(10, 4))
plt.subplot(3, 1, 1)
plt.stem(x, use_line_collection=True)
plt.title('Signal x[n]')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(h, use_line_collection=True)
plt.title('Impulse Response h[n]')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(y, use_line_collection=True)
plt.title('Convolution y[n]')
plt.grid(True)

plt.tight_layout()
plt.show()


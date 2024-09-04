import matplotlib.pyplot as plt
import numpy as np  

def dft(s):
    N = len(s)
    X = [0] * N 
    for k in range(N):
        X[k] = sum(s[n] * np.exp(-1j * 2 * np.pi * k * n / N) for n in range(N))
    return X

def idft(X):
    N = len(X)
    s = [0] * N 
    for n in range(N):
        s[n] = sum(X[k] * np.exp(1j * 2 * np.pi * k * n / N) for k in range(N))
    s = [x / N for x in s]
    return s

def circ_shift(x, shift):
    N = len(x)
    shift = shift % N
    return x[-shift:] + x[:-shift]

def circ_conv(x1, x2):
    N = len(x1)
    y = [0] * N 
    for n in range(N):
        y[n] = sum(x1[m] * x2[(n - m) % N] for m in range(N))
    return y

x1 = list(map(float, input("Enter first signal x1 (space-separated values): ").split()))
x2 = list(map(float, input("Enter second signal x2 (space-separated values): ").split()))

N = max(len(x1), len(x2))

while len(x1) < N:
    x1.append(0)

while len(x2) < N:
    x2.append(0)

ytime = circ_conv(x1, x2)

X1 = dft(x1)
X2 = dft(x2)

Yfreq = [X1[k] * X2[k] for k in range(N)]

yfreq = idft(Yfreq)

print("Circular Convolution (Time Domain):", ytime)
print("Circular Convolution (Frequency Domain):", np.real(yfreq))

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.stem(x1)
plt.title('Signal x1')
plt.grid(False)

plt.subplot(4, 1, 2)
plt.stem(x2)
plt.title('Signal x2')
plt.grid(False)

plt.subplot(4, 1, 3)
plt.stem(ytime)
plt.title('Circular Convolution (Time Domain)')
plt.grid(False)

plt.subplot(4, 1, 4)
plt.stem(np.real(yfreq))
plt.title('Circular Convolution (Frequency Domain)')
plt.grid(False)

plt.tight_layout()
plt.show()

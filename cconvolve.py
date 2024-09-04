import numpy as np
import matplotlib.pyplot as plt

# Circular shift function
def circ_shift(x, m):
    n = len(x)  
    m = m % n   
    y = x[n-m:] + x[:n-m] 
    return y  

def cir_conv(x1, x2):
    N = len(x1)
    
    # Time-reverse x2
    x2_rev = x2[::-1]
    
  
    y = np.zeros(N)
    
  
    for n in range(N):
        shifted_x2 = circ_shift(x2_rev, n)  
        y[n] = sum(x1[m] * shifted_x2[m] for m in range(N))
    
    return y

# Example usage
x1 = [1, 2, 3]
x2 = [4, 5, 6]
y = cir_conv(x1, x2)
print(y) 
plt.plot(y)
plt.show() 


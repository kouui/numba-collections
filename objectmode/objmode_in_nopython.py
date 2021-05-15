from numba import njit
import numpy as np

@njit()
def compute_fft(x):
   y = np.zeros(., dtype=np.complex128) 
   with objmode(y='type[:]'):
      y = np.fft.fft(x)
   return y

@njit()
def main():
   #...
   x = np.random.randint(100)
   fft_x = compute_fft(x) 
   #...

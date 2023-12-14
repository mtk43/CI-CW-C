import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np

# Define filepath
filepath = r'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat'

# Define the sampling rate
fs = 25000

# Import data and extract columns
mat = spio.loadmat(filepath, squeeze_me=True)
d = mat['d']

# Compute the real-input FFT
fft_result = np.fft.rfft(d)
frequencies = np.fft.rfftfreq(len(d), d=1/fs)  # Frequency values corresponding to the FFT result

print(len(frequencies[1:]))

# Plot the frequency spectrum excluding the DC component
plt.plot(frequencies[1:], np.abs(fft_result[1:]))
plt.title('Frequency Spectrum of d (Excluding DC Component)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim((0, fs/2))
plt.grid()
plt.show()

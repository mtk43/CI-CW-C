import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Define function to create a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Define filepath
filepath = r'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat'

# Import data and extract columns
mat = spio.loadmat(filepath, squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']

# Specify the range for plotting d
start_index = 1000000
end_index = 1010000
d_plt = d[start_index:end_index]
index_plt = range(start_index, end_index)  # Create the corresponding index range

# Low-pass filter parameters
sampling_rate = 25000  # sampling rate (Hz)
cutoff_frequencies = [1200]  # Adjust this list of cutoff frequencies as needed

# Plot original d
plt.plot(index_plt, d_plt, label='Original d', linewidth=0.5)

# Iterate over cutoff frequencies and plot filtered signals
for cutoff_frequency in cutoff_frequencies:
    # Apply the low-pass filter to the signal
    filtered_d = butter_lowpass_filter(d_plt, cutoff_frequency, sampling_rate)
    
    # Plot low-pass filtered d for the current cutoff frequency
    plt.plot(index_plt, filtered_d, label=f'Low-Pass Filtered d (Cutoff: {cutoff_frequency} Hz)', linewidth=0.5)

# Filter Index values within the specified range
filtered_indices = [idx for idx in Index if start_index <= idx <= end_index]

# Plot the spike indices within the specified range
dot_values = [d[idx] for idx in filtered_indices]
plt.scatter(filtered_indices, dot_values, color='red', marker='o', label='Dot Indices', linewidth=0.5)

# Show the plot
plt.legend()
plt.grid()
plt.show()

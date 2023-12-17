"""Function to simply find the peaks of the data sets D2-D6"""

import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import statistics

# Define function to create a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# File number to be analysed
file_number = 2

# Mapping file numbers to parameters
file_params = {
    2: (1200, 'green', 1),
    3: (1200, 'green', 1),
    4: (1200, 'green', 1.5),
    5: (1000, 'yellow', 3.5),
    6: (800, 'yellow', 3.5),
}

# Validate file number
if file_number not in file_params:
    print('Invalid file number chosen\n')
    exit()

# Extract parameters from the dictionary
cutoff_frequency, c, mult = file_params[file_number]

# Define filepath
filepath = rf'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D{file_number}.mat'

# Import data and extract columns
mat = spio.loadmat(filepath, squeeze_me=True)
d = mat['d']

# Specify the range for plotting d
start_index = 1000000
end_index = 1005000
d_plt = d[start_index:end_index]
index_plt = range(start_index, end_index)  # Create the corresponding index range

# Low-pass filter parameters
sampling_rate = 25000  # sampling rate (Hz)

# Plot original d
plt.plot(index_plt, d_plt, label='Original d', linewidth=0.5)

# Apply the low-pass filter to the signal
filtered_d = butter_lowpass_filter(d_plt, cutoff_frequency, sampling_rate)
    
# Plot low-pass filtered d for the current cutoff frequency
plt.plot(index_plt, filtered_d, label=f'Low-Pass Filtered d (Cutoff: {cutoff_frequency} Hz)', linewidth=0.5)

# Calculate and print the standard deviation of 'd'
std_deviation = statistics.stdev(filtered_d)
print(f"Standard Deviation of 'd': {std_deviation}")

# Define the prominence as a variable
p = std_deviation

# Find peaks in the filtered signal
peaks, _ = find_peaks(filtered_d, prominence=p*mult)

# Convert index_plt to a NumPy array
index_plt_array = np.array(index_plt)

# Plot detected peaks
plt.plot(index_plt_array[peaks], filtered_d[peaks], 'x', color=c, label='Detected Peaks')

# Plot the standard deviation on the axes
plt.axhline(std_deviation, color='red', linestyle='--', label=f'Standard Deviation: {std_deviation:.2f}')

# Show the plot
plt.legend()
plt.grid()
plt.show()
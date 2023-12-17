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

# Define filepath
filepath = r'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat'

# Import data and extract columns
mat = spio.loadmat(filepath, squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']

# Sort the indices and rearrange the corresponding elements in Class
sorted_indices = np.argsort(Index)
Index = Index[sorted_indices]
Class = Class[sorted_indices]

# Low-pass filter parameters
sampling_rate = 25000    # sampling rate (Hz)
cutoff_frequency = 1200  # Adjust this list of cutoff frequencies as needed

# Apply the low-pass filter to the entire signal
filtered_d = butter_lowpass_filter(d, cutoff_frequency, sampling_rate)
    
# Calculate and print the standard deviation of 'd'
std_deviation = statistics.stdev(d)
print(f"Standard Deviation of 'd': {std_deviation}")

# Define promincence for peak detection
p = std_deviation

# Find peaks in the filtered signal
peaks, _ = find_peaks(filtered_d, prominence=p)

# Test 1: Check if correct no. of peaks were found (Recall)
if len(peaks) == len(Index):
    print('Code finds the correct no. of peaks\n')
else:
    # Calculate how many of peaks it found
    found_peaks_pc = (len(peaks)/len(Index)) * 100
    print(f"Code found {round(found_peaks_pc, 2)}% of peaks\n")

# Initialise a results matrix the length of the peaks value
results = np.zeros((len(peaks), 1))

# Test 2: Check if found peaks are correct (precision)
for i in range(len(peaks)):
    # Get the value of the current peak
    peak_value = filtered_d[peaks[i]]  
    
    # Find the closest Index value to the current peak
    closest_index = min(Index, key=lambda x: abs(x - peaks[i]))
    
    # Calculate the distance between the peak and the closest Index value
    distance = abs(peaks[i] - closest_index)
    
    # Check if the distance is less than 50
    results[i] = 1 if distance<=50 else 0

# Calculate and print "score" of peak finding algorithm
score = (sum(results) / len(peaks)) * 100
print(f"Score: {round(score[0], 2)}%\n")
    
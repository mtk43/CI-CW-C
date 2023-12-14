import scipy.io as spio
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

"""Function to run through all the file, find the indices of the peaks and create random classes"""

# Function to create a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Low-pass filter parameters
sampling_rate = 25000  # sampling rate (Hz)
cutoff_frequency = 800  # Adjust this list of cutoff frequencies as needed

# Create arrays to store results
Index = []
Class = []

# Loop through files D2.mat to D6.mat
for file_number in range(6, 7):

    Index = []
    Class = []

    # Define filepath
    filepath = rf'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D{file_number}.mat'

    # Import data and extract columns
    mat = spio.loadmat(filepath, squeeze_me=True)
    d = mat['d']

    # Apply the low-pass filter to the entire signal
    filtered_d = butter_lowpass_filter(d, cutoff_frequency, sampling_rate)

    # Find peaks in the entire filtered signal
    peaks, _ = find_peaks(filtered_d, prominence=4)

    # Store results for each file
    Index.append(peaks)
    Class.append(np.random.randint(1, 5, size=len(peaks)))  # Generating random class values between 1 and 5

    # Save results to a .mat file with column names "Index" and "Class"
    results_filepath = rf'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Code\D{file_number}.mat'
    spio.savemat(results_filepath, {'Index': Index, 'Class': Class, 'd': d})

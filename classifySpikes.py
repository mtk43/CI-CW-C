import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Function to create a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Low-pass filter parameters
sampling_rate = 25000  # sampling rate (Hz)
cutoff_frequency = 1200  # Adjust this list of cutoff frequencies as needed

# Create arrays to store results
Index = []
Class = []

# Loop through files D2.mat to D6.mat
for file_number in range(3, 7):

    Index = []
    Class = []

    # Define filepath
    filepath = rf'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D{file_number-1}.mat'

    # Import data and extract columns
    mat = spio.loadmat(filepath, squeeze_me=True)
    d = mat['d']
    #Index = mat['Index']
    #Class = mat['Class']

    # Sort the indices and rearrange the corresponding elements in Class
    #sorted_indices = np.argsort(Index)
    #Index = Index[sorted_indices]
    #Class = Class[sorted_indices]

    # Plot original d
    #plt.plot(index_range, d[start_index:end_index], label=f'Original d (D{file_number})', linewidth=0.5)

    # Apply the low-pass filter to the entire signal
    filtered_d = butter_lowpass_filter(d, cutoff_frequency, sampling_rate)

    # Plot low-pass filtered d for the current cutoff frequency
    #plt.plot(index_range, filtered_d[start_index:end_index], label=f'Low-Pass Filtered d (D{file_number}, Cutoff: {cutoff_frequency} Hz)', linewidth=0.5)

    # Find peaks in the entire filtered signal
    peaks, _ = find_peaks(filtered_d, height=0)

    # Filter Index values within the specified range
    #filtered_indices = [idx for idx in Index if start_index <= idx <= end_index]

    # Plot the spike indices within the specified range
    #dot_values = [d[idx] for idx in filtered_indices]
    #plt.scatter(filtered_indices, dot_values, color='red', marker='o', label=f'Dot Indices (D{file_number})', linewidth=0.5)

    #index_plt_array = np.array(index_range)

    # Plot detected peaks
    #plt.plot(index_plt_array[peaks[peaks <= len(index_plt_array)]], filtered_d[peaks[peaks <= len(index_plt_array)]], 'x', color='green', label=f'Detected Peaks (D{file_number})')

    # Draw vertical lines at each detected peak with a window of +/- 50 samples
    """window_size = 50
    for peak in peaks:
        if start_index <= index_plt_array[peak] <= end_index:
            # Ensure window indices are within the valid range
            window_start = max(0, peak - window_size)
            window_end = min(len(index_plt_array) - 1, peak + window_size)
            
            plt.axvline(index_plt_array[peak], color='orange', linestyle='--', linewidth=0.5)  # Line at the peak
            plt.axvline(index_plt_array[window_start], color='orange', linestyle='--', linewidth=0.5)  # Line 50 samples before
            plt.axvline(index_plt_array[window_end], color='orange', linestyle='--', linewidth=0.5)  # Line 50 samples after
    """
    # Store results for each file
    Index.append(peaks)
    Class.append(np.random.randint(1, 5, size=len(peaks)))  # Generating random class values between 1 and 5

    #plt.grid()
    #plt.legend()
    #plt.show()

    # Save results to a .mat file
    results_filepath = rf'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Code\D{file_number-1}.mat'
    spio.savemat(results_filepath, {'all_indices': Index, 'all_classes': Class})

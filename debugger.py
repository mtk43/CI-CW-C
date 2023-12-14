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

# Specify the range for plotting d
start_index = 100450 #1000000
end_index = 100800 #1005000
d_plt = d[start_index:end_index]
index_plt = range(start_index, end_index)  # Create the corresponding index range

# Low-pass filter parameters
sampling_rate = 25000  # sampling rate (Hz)
cutoff_frequency = 1200  # Adjust this list of cutoff frequencies as needed

# Plot original d
plt.plot(index_plt, d_plt, label='Original d', linewidth=0.5)

# Apply the low-pass filter to the entire signal
filtered_d = butter_lowpass_filter(d, cutoff_frequency, sampling_rate)

# Plot low-pass filtered d for the current cutoff frequency
plt.plot(index_plt, filtered_d[start_index:end_index], label=f'Low-Pass Filtered d (Cutoff: {cutoff_frequency} Hz)', linewidth=0.5)

# Find peaks in the entire filtered signal
peaks, _ = find_peaks(filtered_d, height=0)
#peaks, _ = find_peaks(d, height=0)

# Filter Index values within the specified range
filtered_indices = [idx for idx in Index if start_index <= idx <= end_index]

# Plot the spike indices within the specified range
dot_values = [d[idx] for idx in filtered_indices]
plt.scatter(filtered_indices, dot_values, color='red', marker='o', label='Dot Indices', linewidth=0.5)

index_plt_array = np.array(index_plt)

# Plot detected peaks
plt.plot(index_plt_array[peaks[peaks <= len(index_plt_array)]], filtered_d[peaks[peaks <= len(index_plt_array)]], 'x', color='green', label='Detected Peaks')

# Draw vertical lines at each detected peak with a window of +/- 50 samples
window_size = 50
for peak in peaks:
    if start_index <= index_plt_array[peak] <= end_index:
        # Ensure window indices are within the valid range
        window_start = max(0, peak - window_size)
        window_end = min(len(index_plt_array) - 1, peak + window_size)
        
        plt.axvline(index_plt_array[peak], color='orange', linestyle='--', linewidth=0.5)  # Line at the peak
        plt.axvline(index_plt_array[window_start], color='orange', linestyle='--', linewidth=0.5)  # Line 50 samples before
        plt.axvline(index_plt_array[window_end], color='orange', linestyle='--', linewidth=0.5)  # Line 50 samples after

plt.grid()
plt.legend(loc='upper right')

# Show the plot
plt.show()
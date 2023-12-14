import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np

# Define filepath
filepath = r'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat'

# Import data and extract columns
mat = spio.loadmat(filepath, squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']

# Specify the range for plotting d
start_index = 100450 #1000000
end_index = 100800 #1010000
d_plt = d[start_index:end_index]
index_plt = range(start_index, end_index)  # Create the corresponding index range

# Calculate a simple moving average with a window size of 10
window_size = 7
moving_avg = np.convolve(d_plt, np.ones(window_size)/window_size, mode='valid')

# Filter Index values within the specified range
filtered_indices = [idx for idx in Index if start_index <= idx <= end_index]

# Plot the spike indices within the specified range
dot_values = [d[idx] for idx in filtered_indices]
plt.scatter(filtered_indices, dot_values, color='red', marker='o', label='Dot Indices', linewidth=0.5)

# Plot original d and moving average
plt.plot(index_plt, d_plt, label='Original d', linewidth=0.5)
plt.plot(index_plt[window_size-1:], moving_avg, label=f'Moving Average (Window Size: {window_size})', linewidth=0.5)

# Show the plot
plt.legend()
plt.grid()
plt.show()

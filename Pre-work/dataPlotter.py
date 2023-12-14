import scipy.io as spio
import matplotlib.pyplot as plt

# Define filepath
#filepath = r'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat'
filepath = r'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Code\D6.mat'

# Import data and extract columns
mat = spio.loadmat(filepath, squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']

# Specify the range for plotting d
start_index = 1000000    # 100450
end_index = 1010000      # 100800
d_plt = d[start_index:end_index]
index_plt = range(start_index, end_index)  # Create the corresponding index range

# Filter Index values within the specified range
filtered_indices = [idx for idx in Index if start_index <= idx <= end_index]

# Plot the spike indices within the specified range
dot_values = [d[idx] for idx in filtered_indices]
plt.scatter(filtered_indices, dot_values, color='red', marker='o', label='Dot Indices', linewidth=0.5)

# Plot d_plt against its index values
plt.plot(index_plt, d_plt, label='d_plt', linewidth=0.5)

# Show the plot
plt.legend()
plt.grid()
plt.show()

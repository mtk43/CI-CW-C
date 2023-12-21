# This file creates a Convolutional Neural Network (CNN) based on the PyTorch framework
# The network is used for classifying EEG recordings and is trained on dataset D1, which 
# is first filtered with a Butterworth filter, and then the filtered d column is used to
# trian the network, and then it is used to classify and is then used to classify D2-D6 

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as spio
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.signal import butter, filtfilt, find_peaks
import statistics

# Define function to create a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    # Calculate the Nyquist frequency
    nyquist = 0.5 * sampling_rate

    # Normalise the cutoff frequency w/ respect to Nyquist frequency
    normal_cutoff = cutoff_frequency / nyquist

    # Define the butterworth filter and apply it to the input signal
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)

    return y

# Define the convolutional neural network model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class
        super(ConvolutionalNetwork, self).__init__()

        # Define the first two convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Define the two fully connected layers
        self.fc1 = nn.Linear(768, 128)  
        self.fc2 = nn.Linear(128, 5)  # Output classes: 5

    def forward(self, x):
        # Apply the convolutional layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten the tensor for input to the fully connected layers
        x = x.view(-1, 64 * 12)  # Adjust size based on your window size
        
        # Apply the ReLU activation functionand output the result of the second fully connected layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define function to train the network
def train_network(d, indices, Class, window_shift, window_size):
    # Loop through the indices of the spikes
    for i in range(len(indices)):
        # Define the original window
        window_original = np.arange(indices[i] - window_size//2, indices[i] + window_size//2)

        # Loop through the different shifts of the window
        for shift in window_shift:
            # Shift the window
            window = np.roll(window_original, shift)

            # Extract the correct class at that index and convert to a tensor
            target_class = Class[i]
            target_values = torch.tensor(target_class - 1, dtype=torch.long)

            # Extract the d values in the window and conver to a tensor
            x_train_tensor = torch.tensor(d[window], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            y_train_tensor = target_values.unsqueeze(0)

            # Initialise the gradient to zero
            optimizer.zero_grad()

            # compute the predicted class, calculate its errror and update the model parameters
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

# Define function to classify the spikes using the trained CNN
def classifySpikes(d, Index, Class, window_size):
    for i in range(len(Index)):
        # Create a window around each Index and convert array into tensor
        window = range(Index[i] - window_size//2, Index[i] + window_size//2)
        x_test_tensor = torch.tensor(d[window], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Get prediction from network
        outputs = model(x_test_tensor)
        predicted_class = torch.argmax(outputs).item() + 1

        # Place this in the Class list
        Class.append(predicted_class)
    return Class

# Define filepath
filepath = r'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat'

# Import data and extract columns
mat = spio.loadmat(filepath, squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']

# Sort the data in ascending order
sorted_indices = np.argsort(Index)
Index = Index[sorted_indices]
Class = Class[sorted_indices]

# Low-pass filter parameters
sampling_rate = 25000  # sampling rate (Hz)
cutoff_frequency = 1200  

# Apply the low-pass filter to the test signal
d_filt = butter_lowpass_filter(d, cutoff_frequency, sampling_rate)

# Calculate and print the standard deviation of 'd'
std_deviation = statistics.stdev(d_filt)

# Find peaks in the filtered signal
peaks, _ = find_peaks(d_filt, prominence=std_deviation)

# Initilise zero matrices for the index and classes of the detected peaks
index_filt = np.zeros(len(peaks), dtype=int)
class_filt = np.zeros(len(peaks), dtype=int)

# Find the classes corresponding to the filtered peaks
for i in range(len(peaks)):
    # Get the value of the current peak
    peak_value = d[peaks[i]]  
    
    # Find the closest Index value to the current peak
    closest_index = min(Index, key=lambda x: abs(x - peaks[i]))

    # Find the class value associated with the closest_index
    class_value = Class[np.where(Index == closest_index)][0]

    # Fill in the index and class arrays
    index_filt[i] = round(closest_index)
    class_filt[i] = round(class_value)

# Define the window size and learning rate for the CNN
window_size = 50
learning_rate = 0.01

# Create a new instance of the CNN model, and define its loss model and optimisation algorithm
model = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()                           
optimizer = optim.SGD(model.parameters(), learning_rate)    # Stochastic gradient descent

# Define the shifts of the window
window_shift = list(range(-10, 10))

# Train the network
train_network(d, Index, Class, window_shift, window_size)

# Loop through D2-D6, detect spikes and classify them
for file_number in range(2, 7):

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

    # Initialise empty Class & Index variables to store infor for each dataset
    Index = []
    Class = []

    # Define filepath
    filepath = rf'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D{file_number}.mat'

    # Import data and extract columns
    mat = spio.loadmat(filepath, squeeze_me=True)
    d = mat['d']

    # Apply the low-pass filter to the entire signal
    filtered_d = butter_lowpass_filter(d, cutoff_frequency, sampling_rate)

    # Calculate the standard deviation of 'd'
    std_deviation = statistics.stdev(filtered_d)

    # Find peaks in the entire filtered signal
    peaks, _ = find_peaks(filtered_d, prominence=std_deviation*mult)

    # Store results for each file
    Index.append(peaks)

    # Classify each spike
    Class = classifySpikes(filtered_d, peaks, Class, window_size)

    # Save results to a .mat file with column names "Index" and "Class"
    results_filepath = rf'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Code\D{file_number}.mat'
    spio.savemat(results_filepath, {'Index': Index, 'Class': Class})

print('Classificaiton & file generation is complete')
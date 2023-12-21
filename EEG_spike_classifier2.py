# This file creates an Artificial Neural Network (ANN) with two hidden layers based on 
# the PyTorch framework the network is used for classifying EEG recordings and is trained 
# on dataset D1, which is first filtered with a Butterworth filter, and then the filtered 
# d column is used to train the network, and then it is used to classify and is then used to
# classify D2-D6 

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
from scipy.signal import butter, filtfilt, find_peaks

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

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes):
        # Initialise the neural network in the constructor
        super(NeuralNetwork, self).__init__()

        # Define the fully connected hidden layers
        self.fc1 = nn.Linear(input_nodes, hidden1_nodes)
        self.fc2 = nn.Linear(hidden1_nodes, hidden2_nodes)
        self.fc3 = nn.Linear(hidden2_nodes, output_nodes)
        
        # Define the activation function (Logistic sigmoid)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Run the inut through the layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

# Define function to train the network
def train_network(d, indices, Class, window_shift, window_size):
    # Loop through the indices of the spikes
    for i in range(len(indices)):
        # Loop through the different window shifts
        for shift in window_shift:
            # Defint he window position
            start_index = indices[i] + shift
            end_index = start_index + window_size
            window = range(start_index, end_index)

            # Extract the target class
            target_class = Class[i]

            # Create target values for each class
            target_values = [0.01] * output_nodes  # Initialize with small values
            target_values[target_class-1] = 0.99   # Set the target class to 0.99 (Take into account zero index of Python)

            # Convert your NumPy arrays to PyTorch tensors
            x_train_tensor = torch.tensor(d[window], dtype=torch.float32)
            y_train_tensor = torch.tensor(target_values, dtype=torch.float32)

            # Initialise the gradient to zero
            optimizer.zero_grad()

            # Compute the perdicted class, calculate its error and update the model parameters
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

def test_network(d, Index_test, Class_test):
    # Initialise lists for the results
    scorecard = []
    predicted_classes = []
    for i in range(len(Index_test)):
        # Define the window size
        window = range(Index_test[i] - 10, Index_test[i] + 40)

        # Conver the numpy lists to tensors        
        x_test_tensor = torch.tensor(d[window], dtype=torch.float32)
        y_test_tensor = torch.tensor(Class_test[i], dtype=torch.int)

        # Calcualte the outputs
        outputs = model(x_test_tensor)

        # Extract the predicted value
        predicted_class = torch.argmax(outputs).item() + 1
        predicted_classes.append(predicted_class)

        # Determine if they are correct
        if predicted_class == y_test_tensor:
            scorecard.append(1)
        else:
            scorecard.append(0)

    return scorecard, predicted_classes

# Define function to classify the spikes using the trained ANN
def classifySpikes(d, Index, Class, window_size):
    for i in range(len(Index)):
        # Create a window around each Index and convert array into tensor
        window = range(Index[i] - 10, Index[i] + 40)
        x_test_tensor = torch.tensor(d[window], dtype=torch.float32)
        
        # Get prediction from network
        outputs = model(x_test_tensor)
        predicted_class = torch.argmax(outputs).item() + 1

        # Place this in the Class list
        Class.append(predicted_class)
    return Class

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

# Find peaks in the filtered signal
peaks, _ = find_peaks(d_filt, prominence=1)

# Initialise empty arrays for the indeces and classes of the filtered d signal
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

    index_filt[i] = round(closest_index)
    class_filt[i] = round(class_value)

# Define the first 80% of the dataset into a training set
Index_train = Index[:round(len(index_filt) * 0.8)]
Class_train = Class[:round(len(index_filt) * 0.8)]

# Define the last 20% of the dataset as a testing set
Index_test = Index[round(len(index_filt) * 0.8) + 1:]
Class_test = Class[round(len(index_filt) * 0.8) + 1:]

# Initialise model parameters
window_size = 50
input_nodes = window_size
hidden1_nodes = 70
hidden2_nodes = hidden1_nodes
output_nodes = 5
learning_rate = 0.2
epochs = 12
window_Size = 50

# Instantiate the neural network model
model = NeuralNetwork(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Define the shifts of the window
window_shift = [-15, -12, -10, -7, -5, 0, 5, 7, 10, 12, 15]

# Train the network
train_network(d_filt, Index_train, Class_train, window_shift, window_size)

# Test the network on tesitng set to ensure accuracy
scorecard, predicted_classes = test_network(d_filt, Index_test, Class_test)

# Calculate the final percentage performance
scorecard_array = np.array(scorecard)
performance = (scorecard_array.sum() / scorecard_array.size) * 100.0

# Print the performance
print(f"Performance: {performance:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Class_test, predicted_classes)

# Plot confusion matrix
class_names = [str(i) for i in range(1, 6)]
plot_confusion_matrix(cm, classes=class_names, normalize=False)

# Loop through D2-D6, detect spikes and classify them
for file_number in range(2, 7):

    # Mapping file numbers to parameters
    file_params = {
    2: (1200, 'green', 1),
    3: (1200, 'green', 1),
    4: (1200, 'green', 2),
    5: (1000, 'yellow', 4),
    6: (800, 'yellow', 4),
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

    # Find peaks in the entire filtered signal
    peaks, _ = find_peaks(filtered_d, prominence=mult)

    # Store results for each file
    Index.append(peaks)

    # Classify each spike
    Class = classifySpikes(filtered_d, peaks, Class, window_size)

    # Save results to a .mat file with column names "Index" and "Class"
    results_filepath = rf'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Code\D{file_number}.mat'
    spio.savemat(results_filepath, {'Index': Index, 'Class': Class})

print('File generation is complete')
plt.show()
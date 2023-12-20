"""Detect and classify EEG spikes from unfiltered dataset D1 using a pytorch CNN'"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
from scipy.signal import butter, filtfilt, find_peaks
import statistics

# Define function to create a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Define the convolutional neural network model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        #self.fc1 = nn.Linear(64 * 25, 128)  # Adjust input size based on your window size
        self.fc1 = nn.Linear(768, 128)  # Adjust input size based on your window size
        self.fc2 = nn.Linear(128, 5)  # Output classes: 5

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        #print(x.size())
        x = x.view(-1, 64 * 12)  # Adjust size based on your window size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define function to train the network
def train_network(d, indices, Class, window_shift, window_size):
    for i in range(len(indices)):
        window_original = np.arange(indices[i] - window_size//2, indices[i] + window_size//2)
        for shift in window_shift:
            #start_index = indices[i] + shift
            #end_index = start_index + window_size
            #window = range(start_index, end_index)
            window = np.roll(window_original, shift)

            target_class = Class[i]
            target_values = torch.tensor(target_class - 1, dtype=torch.long)

            x_train_tensor = torch.tensor(d[window], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            #y_train_tensor = target_values
            y_train_tensor = target_values.unsqueeze(0)

            
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

def test_network(d, Index_test, Class_test, window_size):
    scorecard = []
    predicted_classes = []
    for i in range(len(Index_test)):
        #window = range(Index_test[i] - 10, Index_test[i] + 40)
        window = range(Index_test[i] - window_size//2, Index_test[i] + window_size//2)
        x_test_tensor = torch.tensor(d[window], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        #y_test_tensor = torch.tensor(Class_test[i] - 1, dtype=torch.long)

        outputs = model(x_test_tensor)
        predicted_class = torch.argmax(outputs).item() + 1
        predicted_classes.append(predicted_class)

        if predicted_class == Class_test[i]:
            scorecard.append(1)
        else:
            scorecard.append(0)

    return scorecard, predicted_classes

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
    plt.show()

# Define filepath
filepath = r'C:\Users\maxno\OneDrive - University of Bath\Documents\Year 5\Semester 1\Computational Intelligence\CW\C\Datasets\D1.mat'

# Import data and extract columns
mat = spio.loadmat(filepath, squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']

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

Index_train = Index[:round(len(Index) * 0.8)]
Class_train = Class[:round(len(Class) * 0.8)]

Index_test = Index[round(len(Index) * 0.8) + 1:]
Class_test = Class[round(len(Class) * 0.8) + 1:]

window_size = 50
learning_rate = 0.01

model = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), learning_rate)

#window_shift = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
window_shift = list(range(-10, 10))

# Train the network
train_network(d, Index_train, Class_train, window_shift, window_size)


# Test the network
scorecard, predicted_classes = test_network(d, Index_test, Class_test, window_size)

# Calculate the final percentage
scorecard_array = np.array(scorecard)
performance = (scorecard_array.sum() / scorecard_array.size) * 100.0

# Print the performance
print(f"Performance: {performance:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Class_test, predicted_classes)
#cm = confusion_matrix(class_test_filt, predicted_classes)

# Plot confusion matrix
class_names = [str(i) for i in range(1, 6)]
plot_confusion_matrix(cm, classes=class_names, normalize=False)

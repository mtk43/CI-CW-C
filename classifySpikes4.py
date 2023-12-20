"""Detect and classify EEG spikes from unfiltered dataset D1 suing a pytorch ANN"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import statistics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden1_nodes)
        self.fc2 = nn.Linear(hidden1_nodes, hidden2_nodes)
        self.fc3 = nn.Linear(hidden2_nodes, output_nodes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

# Define function to train the network
def trainNetwork(d, indices, Class, window_shift, window_size):
    for i in range(len(indices)):
        for shift in window_shift:
            # Calculate the start and end indices for the window
            start_index = indices[i] + shift
            end_index = start_index + window_size

            # Create a window around each Index
            window = range(start_index, end_index)

            # Extract the target class
            target_class = Class[i]

            # Create target values for each class
            target_values = [0.01] * output_nodes  # Initialize with small values
            target_values[target_class-1] = 0.99   # Set the target class to 0.99 (Take into account zero index of Python)

            # Convert your NumPy arrays to PyTorch tensors
            x_train_tensor = torch.tensor(d[window], dtype=torch.float32)
            y_train_tensor = torch.tensor(target_values, dtype=torch.float32)           
            
            # Create DataLoader for training
            #train_data = TensorDataset(x_train_tensor, y_train_tensor)
            #train_loader = DataLoader(train_data, batch_size=1, shuffle=True) 
            
            # Zero the parameter gradient
            optimizer.zero_grad()

            # Forward -> Backward -> Optimise
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

def testNetwork(d, Index_test, Class_test, window_size):
    scorecard = []
    predicted_classes = []
    for i in range(len(Index_test)):
        # Create a window around each Index (remember range function is not inclusive)
        window = range(Index_test[i]-10, Index_test[i] + 40)

        x_test_tensor = torch.tensor(d[window], dtype=torch.float32)
        y_test_tensor = torch.tensor(Class_test[i], dtype=torch.int)

        # Calculate the model output
        outputs = model(x_test_tensor)

        predicted_class = torch.argmax(outputs).item() + 1
        predicted_classes.append(predicted_class)

        # Append either a 1 or a 0 to the scorecard list
        if predicted_class == y_test_tensor:
            scorecard.append(1)
        else:
            scorecard.append(0)
       
    return scorecard, predicted_classes


# Define function to plot confusion matrix
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

# Sort the indices and rearrange the corresponding elements in Class
sorted_indices = np.argsort(Index)
Index = Index[sorted_indices]
Class = Class[sorted_indices]

# Sort the variables into a training and testing set
#d_train = d[0:round(len(d)*0.8)]
Index_train = Index[:round(len(Index)*0.8)]
Class_train = Class[:round(len(Class)*0.8)]

#d_test = d[round(len(d)*0.8)+1:-1]
Index_test = Index[round(len(Index)*0.8)+1:]
Class_test = Class[round(len(Class)*0.8)+1:]

# Define the window size
window_size = 50

# Convert your NumPy arrays to PyTorch tensors
"""X_train_tensor = torch.tensor(d[range(Index_train[0]-10, Index_train[0] + 40)], dtype=torch.float32)
y_train_tensor = torch.tensor(Class_train[0], dtype=torch.float32)

X_test_tensor = torch.tensor(d[range(Index_test[0]-10, Index_test[0] + 40)], dtype=torch.float32)
y_test_tensor = torch.tensor(Class_test[0], dtype=torch.float32)

# Create DataLoader for training
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)"""

# Instantiate the neural network model
input_nodes = window_size
hidden1_nodes = 70
hidden2_nodes = hidden1_nodes
output_nodes = 5
learning_rate = 0.2
epochs = 8

model = NeuralNetwork(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Define a window shift array
#window_shift = [-15, -12, -10, -7, -5, 0, 5, 7, 10, 12, 15]
window_shift = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
#window_shift = range(-10, 11)

# Train the network
trainNetwork(d, Index_train, Class_train, window_shift, window_size)

# Train the model
"""for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()"""

scorecard = []
predicted_classes = []
# Test the model
"""with torch.no_grad():
    X_test_tensor = torch.tensor(d[range(Index_test[0]-10, Index_test[0] + 40)], dtype=torch.float32)
    y_test_tensor = torch.tensor(Class_test[0], dtype=torch.long)

    outputs = model(X_test_tensor)

    predicted_class = torch.argmax(outputs).item() + 1
    predicted_classes.append(predicted_classes)

    # Append either a 1 or a 0 to the scorecard list
    if predicted_classes == y_test_tensor:
        scorecard.append(1)
    else:
        scorecard.append(0)"""

# Test the network
scorecard, predicted_classes = testNetwork(d, Index_test, Class_test, window_size)

# Calculate the final percentage
scorecard_array = np.array(scorecard)
performance = (scorecard_array.sum() / scorecard_array.size) * 100.0

# Print the performance
print(f"Performance: {performance:.2f}%")

# Create confusion matrix
cm = confusion_matrix(Class_test, predicted_classes)

# Plot confusion matrix
class_names = [str(i) for i in range(1, output_nodes + 1)]
plot_confusion_matrix(cm, classes=class_names, normalize=False)

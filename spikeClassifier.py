"""Detect and classify EEG spikes from datasets D2-D6"""

import ANN 
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import statistics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools

# Define function to create a low-pass Butterworth filter
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Define a function to train the ANN
def trainNetwork(d, indices, Class):
    # Train the neural network on each training sample
    for i in range(len(indices)):
        # Create a window around each Index 
        window = range(indices[i], indices[i] + 30)

        # Extract the d values in the window
        input_data = d[window]

        # Extract the target class
        target_class = Class[i]

        # Create target values for each class
        target_values = [0.01] * outputNodes  # Initialize with small values
        target_values[target_class-1] = 0.99  # Set the target class to 0.99 (Take into account zero index of Python)

        # Train the network
        N.train(input_data, target_values)

#Define function to test the ANN
def testNetwork(d, Index_test, Class_test):
    scorecard = []

    for i in range(len(Index_test)):
        # Create a window around each Index (remember range function is not inclusive)
        window = range(Index_test[i], Index_test[i] + 30)

        # Extract the d values in the window
        input_data = d[window]

        # Extract the target class
        correct_class = Class_test[i]

        # Query the network
        outputs = N.query(input_data)

        # The index of the highest value output corresponds to the predicted class
        predicted_class = np.argmax(outputs)

        # Append either a 1 or a 0 to the scorecard list
        if predicted_class == correct_class:
            scorecard.append(1)
        else:
            scorecard.append(0)
    return scorecard

# Function to plot confusion matrix
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

#Declaring the ANN properties
hiddenNodes = 1400
learningRate= 0.1
epochs = 4
inputNodes = 30 # Corresponds to the input window size(?)
outputNodes = 5 # Corresponds to no. of classifications

            
# Define the ANN 
N = ANN.NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

# Train the network for the defined no. of epochs
iteration = 1
while iteration <= epochs:
    trainNetwork(d, Index_train, Class_train)
    iteration += 1

# Test the network and create its scorecard
scorecard = testNetwork(d, Index_test, Class_test)

# Calculate the final percentage
scorecard_array = np.array(scorecard)
performance = (scorecard_array.sum() / scorecard_array.size) * 100.0

# Print the performance
print(f"Performance: {performance:.2f}%")

# Get the predictions
predictions = [np.argmax(N.query(d[range(i, i + 30)])) + 1 for i in Index_test]

# Create confusion matrix
cm = confusion_matrix(Class_test, predictions)

# Plot confusion matrix
class_names = [str(i) for i in range(1, outputNodes + 1)]
#plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')
plot_confusion_matrix(cm, classes=class_names, normalize=False)

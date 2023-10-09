#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatalabel.xlsx")
df


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Separate features (X) and target (y)
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']].values  # Convert DataFrame columns to a NumPy array
y = binary_df['Label'].values

# Split the data into training and test sets (70% training, 30% test)
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Initialize weights
weights = np.array([W0, W1, W2])

# Initialize the number of epochs and convergence condition
max_epochs = 1000
convergence_error = 0.002

# Lists to store epoch and error values for plotting
epoch_list = []
error_list = []

# Training loop
for epoch in range(max_epochs):
    error_sum = 0
    for i in range(len(Tr_X)):
        # Compute the weighted sum
        weighted_sum = np.dot(weights, np.insert(Tr_X[i], 0, 1))
        # Apply the step activation function
        prediction = step_activation(weighted_sum)
        # Compute the error
        error = Tr_y[i] - prediction
        error_sum += error
        # Update the weights
        weights += learning_rate * error * np.insert(Tr_X[i], 0, 1)
    
    # Calculate the sum-squared error
    mse = (error_sum ** 2) / len(Tr_X)
    
    # Append epoch and error values to lists for plotting
    epoch_list.append(epoch + 1)
    error_list.append(mse)
    
    # Print the error for this epoch
    print(f"Epoch {epoch + 1}: Mean Squared Error = {mse}")
    
    # Check for convergence
    if mse <= convergence_error:
        print("Convergence reached. Training stopped.")
        break

# Print the final weights
print("Final Weights:")
print(f"W0 = {weights[0]}, W1 = {weights[1]}, W2 = {weights[2]}")

# Plot epochs against error values
plt.figure(figsize=(8, 6))
plt.plot(epoch_list, error_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Epochs vs. Error")
plt.grid(True)
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

 

# Define the initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

 

# Load your data from a DataFrame (binary_df)
# Assuming 'embed_1' and 'embed_2' are your input features, and 'Label' is your target label.
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']

 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

 

# Initialize variables for tracking epochs and errors
epochs = 0
errors = []

 

# Define activation functions
def bipolar_step(x):
    return -1 if x < 0 else 1

 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

 

def relu(x):
    return max(0, x)

 

# Select the activation function to use
activation_function = sigmoid  # Change this to bipolar_step or relu for different activations

 

# Training the perceptron
max_epochs = 1000  # Set a maximum number of epochs to avoid infinite looping

 

while epochs < max_epochs:
    total_error = 0

 

    for i in range(0, len(X_train)):
        # Calculate the weighted sum of inputs
        weighted_sum = W0 + W1 * X_train.iloc[i]['embed_1'] + W2 * X_train.iloc[i]['embed_2']

 

        # Apply the selected activation function
        prediction = activation_function(weighted_sum)

 

        # Calculate the error
        error = y_train.iloc[i] - prediction

 

        # Update weights
        W0 = W0 + learning_rate * error
        W1 = W1 + learning_rate * error * X_train.iloc[i]['embed_1']
        W2 = W2 + learning_rate * error * X_train.iloc[i]['embed_2']

 

        total_error += error ** 2

 

    # Append the total error for this epoch to the list
    errors.append(total_error)

 

    # Check for convergence
    if total_error == 0:
        break

 

    epochs += 1

 

# Plot epochs against error values
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error in Perceptron Training')
plt.show()

 

# Test the trained perceptron on the test data
correct_predictions = 0

 

for i in range(len(X_test)):
    weighted_sum = W0 + W1 * X_test.iloc[i]['embed_1'] + W2 * X_test.iloc[i]['embed_2']

 

    # Apply the selected activation function
    prediction = activation_function(weighted_sum)

 

    if prediction == y_test.iloc[i]:
        correct_predictions += 1

 

accuracy = correct_predictions / len(X_test)

 

print(f"Activation Function: {activation_function.__name__}")
print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")
print(f"Final Weights: W0 = {W0}, W1 = {W1}, W2 = {W2}")
print(f"Number of Epochs: {epochs}")


# In[ ]:

#A1
import numpy as np

# Define initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Training data for AND gate
# AND gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

def activate(sum):
    return 1 if sum >= 0 else 0

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Calculate the error
            error = target - activate(weighted_sum)
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Train the perceptron and collect errors
trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs)

# Print the trained weights
print("Trained Weights:")
print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

# Test the perceptron
def test_perceptron(weights, data):
    correct = 0
    for i in range(len(data)):
        x1, x2 = data[i]
        target = targets[i]
        weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
        prediction = activate(weighted_sum)
        if prediction == target:
            correct += 1
        print(f"Input: ({x1}, {x2}), Target: {target}, Prediction: {prediction}")
    accuracy = (correct / len(data)) * 100
    print(f"Accuracy: {accuracy}%")

# Test the trained perceptron
print("\nTesting the Trained Perceptron:")
test_perceptron(trained_weights, inputs)

# Plot epochs against error values
import matplotlib.pyplot as plt
plt.plot(range(len(error_values)), error_values)
plt.xlabel('Epochs')
plt.ylabel('Sum-Square-Error')
plt.title('Epochs vs. Sum-Square-Error in Perceptron Training')
plt.show()


#A2
import numpy as np

# Define initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Training data for AND gate
# AND gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

# Activation functions
def bi_polar_step_activation(sum):
    return -1 if sum < 0 else 1

def sigmoid_activation(sum):
    return 1 / (1 + np.exp(-sum))

def relu_activation(sum):
    return max(0, sum)

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data, activation_fn):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Apply the selected activation function
            activated_sum = activation_fn(weighted_sum)
            # Calculate the error
            error = target - activated_sum
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Test different activation functions and compare iterations
activation_functions = [("Bi-Polar Step", bi_polar_step_activation),
                        ("Sigmoid", sigmoid_activation),
                        ("ReLU", relu_activation)]

for activation_name, activation_fn in activation_functions:
    print(f"\nTraining with {activation_name} Activation Function:")
    trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs, activation_fn)
    print("Trained Weights:")
    print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

    print("\nTesting the Trained Perceptron:")
    test_perceptron(trained_weights, inputs)
    print(f"Number of Iterations to Converge: {len(error_values)}")

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(error_values) + 1), error_values)
    plt.xlabel('Epochs')
    plt.ylabel('Total Error')
    plt.title(f'Error Convergence ({activation_name} Activation Function)')
    plt.grid(True)
    plt.show()


#A3
import numpy as np
import matplotlib.pyplot as plt

# Define initial weights
W0 = 10
W1 = 0.2
W2 = -0.75

# Training data for AND gate
# AND gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

def activate(sum):
    return 1 if sum >= 0 else 0

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Calculate the error
            error = target - activate(weighted_sum)
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return errors

# Varying learning rates
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
iteration_counts = []

for lr in learning_rates:
    # Clone the initial weights to keep them the same for each learning rate
    weights = [W0, W1, W2]
    # Train the perceptron and collect errors
    error_values = train_perceptron(weights, lr, 100, inputs)
    iteration_counts.append(len(error_values))

# Plot the number of iterations vs. learning rates
plt.plot(learning_rates, iteration_counts, marker='o', linestyle='-', color='b')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations to Converge')
plt.title('Convergence Analysis with Varying Learning Rates')
plt.grid(True)
plt.show()


#A4
import numpy as np

# Define initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Training data for XOR gate
# XOR gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

def activate(sum):
    return 1 if sum >= 0 else 0

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Calculate the error
            error = target - activate(weighted_sum)
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Train the perceptron and collect errors
trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs)

# Print the trained weights
print("Trained Weights:")
print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

# Test the perceptron
def test_perceptron(weights, data):
    correct = 0
    for i in range(len(data)):
        x1, x2 = data[i]
        target = targets[i]
        weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
        prediction = activate(weighted_sum)
        if prediction == target:
            correct += 1
        print(f"Input: ({x1}, {x2}), Target: {target}, Prediction: {prediction}")
    accuracy = (correct / len(data)) * 100
    print(f"Accuracy: {accuracy}%")

# Test the trained perceptron
print("\nTesting the Trained Perceptron:")
test_perceptron(trained_weights, inputs)


#A4
import numpy as np

# Define initial weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Training data for AND gate
# XOR gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# Activation functions
def bi_polar_step_activation(sum):
    return -1 if sum < 0 else 1

def sigmoid_activation(sum):
    return 1 / (1 + np.exp(-sum))

def relu_activation(sum):
    return max(0, sum)

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data, activation_fn):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Apply the selected activation function
            activated_sum = activation_fn(weighted_sum)
            # Calculate the error
            error = target - activated_sum
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return weights, errors

# Test different activation functions and compare iterations
activation_functions = [("Bi-Polar Step", bi_polar_step_activation),
                        ("Sigmoid", sigmoid_activation),
                        ("ReLU", relu_activation)]

for activation_name, activation_fn in activation_functions:
    print(f"\nTraining with {activation_name} Activation Function:")
    trained_weights, error_values = train_perceptron([W0, W1, W2], learning_rate, 100, inputs, activation_fn)
    print("Trained Weights:")
    print(f"W0: {trained_weights[0]}, W1: {trained_weights[1]}, W2: {trained_weights[2]}")

    print("\nTesting the Trained Perceptron:")
    test_perceptron(trained_weights, inputs)
    print(f"Number of Iterations to Converge: {len(error_values)}")

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(error_values) + 1), error_values)
    plt.xlabel('Epochs')
    plt.ylabel('Total Error')
    plt.title(f'Error Convergence ({activation_name} Activation Function)')
    plt.grid(True)
    plt.show()


#A4
import numpy as np
import matplotlib.pyplot as plt

# Define initial weights
W0 = 10
W1 = 0.2
W2 = -0.75

# Training data for XOR gate
# XOR gate truth table: inputs and corresponding outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

def activate(sum):
    return 1 if sum >= 0 else 0

# Perceptron training function
def train_perceptron(weights, learning_rate, max_epochs, data):
    errors = []  # To store error values for each epoch
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2 = data[i]
            target = targets[i]
            # Calculate the weighted sum
            weighted_sum = weights[0] + weights[1] * x1 + weights[2] * x2
            # Calculate the error
            error = target - activate(weighted_sum)
            total_error += error
            # Update weights
            weights[0] += learning_rate * error
            weights[1] += learning_rate * error * x1
            weights[2] += learning_rate * error * x2
        errors.append(total_error)
        if total_error == 0:
            break
    return errors

# Varying learning rates
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
iteration_counts = []

for lr in learning_rates:
    # Clone the initial weights to keep them the same for each learning rate
    weights = [W0, W1, W2]
    # Train the perceptron and collect errors
    error_values = train_perceptron(weights, lr, 100, inputs)
    iteration_counts.append(len(error_values))

# Plot the number of iterations vs. learning rates
plt.plot(learning_rates, iteration_counts, marker='o', linestyle='-', color='b')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations to Converge')
plt.title('Convergence Analysis with Varying Learning Rates')
plt.grid(True)
plt.show()

#A5
import numpy as np

# Define initial weights and learning rate
W0 = 0.1
W1 = 0.1
W2 = 0.1
W3 = 0.1
learning_rate = 0.1

# Training data
data = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
])

# Target values (High Value or Low Value)
targets = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x1, x2, x3, x4):
    weighted_sum = W0 + W1 * x1 + W2 * x2 + W3 * x3
    return sigmoid(weighted_sum)

def train_perceptron(max_epochs, data, targets):
    global W0, W1, W2, W3  # Declare global variables
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(len(data)):
            x1, x2, x3, x4 = data[i]
            target = targets[i]
            prediction = predict(x1, x2, x3, x4)
            error = target - prediction
            total_error += error
            W0 += learning_rate * error
            W1 += learning_rate * error * x1
            W2 += learning_rate * error * x2
            W3 += learning_rate * error * x3
        if total_error == 0:
            break

# Train the perceptron
train_perceptron(1000, data, targets)

# Test the perceptron and print the results
for i in range(len(data)):
    x1, x2, x3, x4 = data[i]
    prediction = predict(x1, x2, x3, x4)
    classification = "Yes" if prediction >= 0.5 else "No"
    print(f"Customer C_{i + 1}: {classification}")

#A6
# Calculate the pseudo-inverse of the data
pseudo_inverse = np.linalg.pinv(data)
print("Pseudo inverse is",pseudo_inverse)

#A7
import numpy as np

class ANDGateNeuralNetwork:
    def __init__(self, learning_rate=0.05):
        self.weights_ih = np.random.randn(2, 2)
        self.weights_ho = np.random.randn(1, 2)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        h = self.sigmoid(np.dot(self.weights_ih, inputs))
        o = self.sigmoid(np.dot(self.weights_ho, h))
        return o

    def backpropagate(self, inputs, target_output, actual_output, h):
        error = target_output - actual_output

        gradient_ho = error * actual_output * (1 - actual_output)
        gradient_ih = (gradient_ho @ self.weights_ho) * h * (1 - h)

        self.weights_ho += self.learning_rate * np.outer(gradient_ho, h)
        self.weights_ih += self.learning_rate * np.outer(gradient_ih, inputs)

    def train(self, training_examples):
        for inputs, target_output in training_examples:
            h = self.sigmoid(np.dot(self.weights_ih, inputs))
            actual_output = self.forward_propagate(inputs)
            self.backpropagate(inputs, target_output, actual_output, h)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new AND gate neural network
network = ANDGateNeuralNetwork()

# Train the network on the AND gate truth table
training_examples = [(np.array([0, 0]), 0), (np.array([0, 1]), 0), (np.array([1, 0]), 0), (np.array([1, 1]), 1)]
network.train(training_examples)

# Test the network for multiple inputs
inputs_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

for inputs in inputs_list:
    output = network.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")

#A8
import numpy as np

class XORGateNeuralNetwork:
    def __init__(self, learning_rate=0.05):
        self.weights_ih = np.random.randn(2, 2)
        self.weights_ho = np.random.randn(1, 2)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagate(self, inputs):
        h = self.sigmoid(np.dot(self.weights_ih, inputs))
        o = self.sigmoid(np.dot(self.weights_ho, h))
        return o

    def backpropagate(self, inputs, target_output, actual_output, h):
        error = target_output - actual_output

        gradient_ho = error * actual_output * (1 - actual_output)
        gradient_ih = (gradient_ho @ self.weights_ho) * h * (1 - h)

        self.weights_ho += self.learning_rate * np.outer(gradient_ho, h)
        self.weights_ih += self.learning_rate * np.outer(gradient_ih, inputs)

    def train(self, training_examples):
        for inputs, target_output in training_examples:
            h = self.sigmoid(np.dot(self.weights_ih, inputs))
            actual_output = self.forward_propagate(inputs)
            self.backpropagate(inputs, target_output, actual_output, h)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new XOR gate neural network
network = XORGateNeuralNetwork()

# Train the network on the XOR gate truth table
training_examples = [(np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 0)]
network.train(training_examples)

# Test the network for multiple inputs
inputs_list = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]

for inputs in inputs_list:
    output = network.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")


#A9
import numpy as np

class ANDGatePerceptron:
    def __init__(self, learning_rate=0.05):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn(1)
        self.learning_rate = learning_rate

    def forward_propagate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = np.where(weighted_sum >= 0, 1, 0)
        return output

    def backpropagate(self, inputs, target_output, actual_output):
        error = target_output - actual_output
        delta = error * self.learning_rate
        self.weights += delta * inputs
        self.bias += delta

    def train(self, training_examples):
        for inputs, target_output in training_examples:
            actual_output = self.forward_propagate(inputs)
            self.backpropagate(inputs, target_output, actual_output)

    def predict(self, inputs):
        return self.forward_propagate(inputs)

# Create a new perceptron
perceptron = ANDGatePerceptron()

# Create a training dataset
training_examples = [(np.array([0, 0]), np.array([0])), (np.array([0, 1]), np.array([0])), (np.array([1, 0]), np.array([0])), (np.array([1, 1]), np.array([1]))]

# Train the perceptron
perceptron.train(training_examples)

# Test the perceptron
inputs = np.array([1, 1])
output = perceptron.predict(inputs)

# Print the output
print(output)



import numpy as np
from sklearn.neural_network import MLPClassifier

# Create an XOR gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Create an MLPClassifier with one hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=10000, random_state=42)

# Train the model
mlp.fit(X, y)

# Test the model
inputs = np.array([[1, 1]])
output = mlp.predict(inputs)

# Print the output
print(output)


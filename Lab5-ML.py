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





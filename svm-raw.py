import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load the MNIST dataset from CSV
# Replace 'mnist.csv' with the path to your MNIST CSV file
mnist_data = pd.read_csv('raw_reduced.csv')

# Step 2: Separate features and labels
X = mnist_data.iloc[:, :-1].values  # All columns except the last
y = mnist_data.iloc[:, -1].values  # labels

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create an SVM classifier
model = SVC(kernel='linear')  # You can also try 'rbf', 'poly', etc.

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Visualize some predictions
def plot_digits(images, labels, predictions, n=10):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {labels[i]}\nPred: {predictions[i]}')
        plt.axis('off')
    plt.show()

# Visualize the first 10 test images and their predictions
plot_digits(X_test, y_test, y_pred, n=10)

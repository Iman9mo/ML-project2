import pandas as pd
from sklearn.datasets import fetch_openml

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Create a DataFrame from the data
data = pd.DataFrame(mnist.data)
data['label'] = mnist.target  # Add the labels to the DataFrame

# Save the DataFrame to a CSV file
data.to_csv('mnist.csv', index=False)

print("MNIST dataset has been saved to mnist.csv")

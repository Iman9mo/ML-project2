import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def reduce_mnist_data(csv_file, n_components=40, variance_threshold=60):
    # Step 1: Read the MNIST dataset from CSV
    mnist_data = pd.read_csv(csv_file)

    # Extract images and labels
    images = mnist_data.iloc[:, :-1].values  # All columns except the last
    labels = mnist_data.iloc[:, -1].values    # Last column (labels)

    # Step 2: Calculate the mean image and center the images
    mean_image = np.mean(images, axis=0)
    centered_images = images - mean_image  # Centering the images

    # Step 4: Apply PCA to the centered images
    pca = PCA()
    pca.fit(centered_images)

    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio) * 100
    n_component = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f'Number of components to explain at least {variance_threshold}% variance: {n_component}')

    # Plotting
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Variance Explained')
    plt.ylim(0, 110)
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.plot(cumulative_variance)
    plt.show()

    # Reduce the dimensionality of the dataset
    X_reduced = PCA(n_components=n_components).fit_transform(centered_images)
    print(f'Reduced data shape: {X_reduced.shape}')

    # Create a DataFrame for the reduced data and labels
    X_reduced_df = pd.DataFrame(X_reduced)
    X_reduced_df['label'] = labels

    # Save the reduced data to a new CSV file
    output_file = 'reduced_' + csv_file
    X_reduced_df.to_csv(output_file, index=False)
    print(f'Reduced data saved to {output_file}')

# Example usage
reduce_mnist_data('mnist_sobel.csv')
reduce_mnist_data('mnist_gaussian_hog_concatenated.csv')

import pandas as pd
import numpy as np


def convolve2d(image, kernel):
    # Get the dimensions of the kernel
    kernel_height, kernel_width = kernel.shape
    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image to handle borders
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Get the dimensions of the padded image
    padded_height, padded_width = padded_image.shape

    # Create an output array
    output = np.zeros_like(image, dtype=np.float64)

    # Perform convolution
    for i in range(pad_height, padded_height - pad_height):
        for j in range(pad_width, padded_width - pad_width):
            # Extract the region of interest
            region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            # Apply the kernel
            output[i - pad_height, j - pad_width] = np.sum(region * kernel)

    return output


def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel


def apply_gaussian_filter(image, kernel_size=5, sigma=1):
    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    # Apply Gaussian filter
    filtered_image = convolve2d(image, kernel)
    return filtered_image


# Read the MNIST dataset from CSV
mnist_data = pd.read_csv('mnist.csv')

# Extract images and labels
images = mnist_data.iloc[:, :-1].values  # All columns except the last
labels = mnist_data.iloc[:, -1].values  # Last column (labels)

# Prepare an array to hold the processed images
processed_images = []

# Process each image
counter = 0
for image in images:
    # Reshape the flat image into a 28x28 array
    image_reshaped = image.reshape(28, 28)
    # Apply Gaussian filter
    gaussian_image = apply_gaussian_filter(image_reshaped, kernel_size=5, sigma=1)
    # Flatten the processed image and append to the list
    processed_images.append(gaussian_image.flatten())
    print(counter)
    counter += 1

# Convert the processed images to a DataFrame
processed_images_df = pd.DataFrame(processed_images)

# Add labels to the DataFrame
processed_images_df['label'] = labels

# Save the processed images to a new CSV file
processed_images_df.to_csv('mnist_gaussian.csv', index=False)

print("Processed images have been saved to mnist_gaussian.csv")

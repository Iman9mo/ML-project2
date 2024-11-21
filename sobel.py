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


def apply_sobel_filters(image):
    # Sobel kernels
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    # Apply Sobel filters
    gradient_x = convolve2d(image, sobel_x)
    gradient_y = convolve2d(image, sobel_y)

    # Combine gradients (magnitude)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    return gradient_magnitude


# Read the MNIST dataset from CSV
mnist_data = pd.read_csv('mnist.csv')

# Extract images and labels
images = mnist_data.iloc[:, :-1].values  # All columns except the last
labels = mnist_data.iloc[:, -1].values  # Last column (labels)

# Prepare an array to hold the processed images
processed_images = []

# Process each image
counter = 1
for image in images:
    print(counter)
    counter += 1
    # Reshape the flat image into a 28x28 array
    image_reshaped = image.reshape(28, 28)
    # Apply Sobel filters
    sobel_image = apply_sobel_filters(image_reshaped)
    # Flatten the processed image and append to the list
    processed_images.append(sobel_image.flatten())

# Convert the processed images to a DataFrame
processed_images_df = pd.DataFrame(processed_images)

# Add labels to the DataFrame
processed_images_df['label'] = labels

# Save the processed images to a new CSV file
processed_images_df.to_csv('mnist_sobel.csv', index=False)

print("Processed images have been saved to mnist_sobel.csv")

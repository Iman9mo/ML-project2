import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage import exposure


# Function to apply HOG on a single image
def apply_hog(image):
    fd, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        channel_axis=-1
    )

    # Normalize the HOG image for better visualization
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_image, fd  # Return both the HOG image and the feature descriptor


# Read the MNIST dataset from CSV
mnist_data = pd.read_csv('mnist.csv')

# Extract images and labels
images = mnist_data.iloc[:, :-1].values  # All columns except the last
labels = mnist_data.iloc[:, -1].values  # Last column (labels)

# Prepare an array to hold the processed HOG images
hog_images = []

# Process each image
for image in images:
    # Reshape the flat image into a 28x28 array
    image_reshaped = image.reshape(28, 28)
    # Apply HOG
    hog_image, _ = apply_hog(image_reshaped)
    # Flatten the HOG image and append to the list
    hog_images.append(hog_image.flatten())

# Convert the HOG images to a DataFrame
hog_images_df = pd.DataFrame(hog_images)

# Add labels to the DataFrame
hog_images_df['label'] = labels

# Save the HOG images to a new CSV file
hog_images_df.to_csv('mnist_hog.csv', index=False)

print("HOG processed images have been saved to mnist_hog.csv")

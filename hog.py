import pandas as pd
from skimage.feature import hog
from skimage import exposure

# Function to apply HOG on a single image
def apply_hog(image):
    fd, hog_image = hog(
        image,
        orientations=2,
        pixels_per_cell=(4, 4),
        cells_per_block=(1, 1),
        block_norm='L2-Hys',
        visualize=True
    )

    # Normalize the HOG image for better visualization
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_image, fd  # Return both the HOG image and the feature descriptor

# Read the MNIST dataset from CSV
mnist_data = pd.read_csv('mnist.csv')

# Extract images and labels
images = mnist_data.iloc[:, :-1].values  # All columns except the last
labels = mnist_data.iloc[:, -1].values    # Last column (labels)

# Prepare arrays to hold the processed HOG images and feature descriptors
hog_images = []
feature_descriptors = []

# Process each image
counter = 1
for image in images:
    # Reshape the flat image into a 28x28 array
    image_reshaped = image.reshape(28, 28)
    # Apply HOG
    hog_image, fd = apply_hog(image_reshaped)
    # Flatten the HOG image and append to the list
    hog_images.append(hog_image.flatten())
    # Append the feature descriptor to the list
    feature_descriptors.append(fd)
    print(counter)
    counter += 1

# Convert the HOG images to a DataFrame
hog_images_df = pd.DataFrame(hog_images)

# Convert the feature descriptors to a DataFrame
feature_descriptors_df = pd.DataFrame(feature_descriptors)

# Add labels to the HOG images DataFrame
hog_images_df['label'] = labels

# Save the HOG images to a new CSV file
hog_images_df.to_csv('mnist_hog_images.csv', index=False)

# Save the feature descriptors to a new CSV file
feature_descriptors_df.to_csv('mnist_feature_descriptors.csv', index=False)

print("HOG processed images have been saved to mnist_hog_images.csv")
print("Feature descriptors have been saved to mnist_feature_descriptors.csv")

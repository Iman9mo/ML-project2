import pandas as pd

# Step 1: Read the Sobel-filtered images dataset
sobel_data = pd.read_csv('mnist_sobel.csv')

# Step 2: Read the HOG images dataset
hog_data = pd.read_csv('mnist_feature_descriptors.csv')

# Step 3: Concatenate the two datasets
# Ensure that both datasets have the same number of rows (images)
if sobel_data.shape[0] != hog_data.shape[0]:
    raise ValueError("The number of images in the Sobel dataset and HOG dataset must be the same.")

# Concatenate along the columns
concatenated_data = pd.concat([sobel_data.iloc[:, :-1], hog_data], axis=1)  # Exclude labels from sobel data
concatenated_data['label'] = sobel_data['label']
# Step 4: Save the concatenated dataset to a new CSV file
concatenated_data.to_csv('mnist_sobel_hog_concatenated.csv', index=False)

print("Concatenated dataset has been saved to mnist_sobel_hog_concatenated.csv")

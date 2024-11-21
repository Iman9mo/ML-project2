import pandas as pd

# Step 1: Read the gaussian-filtered images dataset
gaussian_data = pd.read_csv('mnist_gaussian.csv')

# Step 2: Read the HOG images dataset
hog_data = pd.read_csv('mnist_feature_descriptors.csv')

# Step 3: Concatenate the two datasets
# Ensure that both datasets have the same number of rows (images)
if gaussian_data.shape[0] != hog_data.shape[0]:
    raise ValueError("The number of images in the gaussian dataset and HOG dataset must be the same.")

# Concatenate along the columns
concatenated_data = pd.concat([gaussian_data.iloc[:, :-1], hog_data], axis=1)  # Exclude labels from gaussian data
concatenated_data['label'] = gaussian_data['label']
# Step 4: Save the concatenated dataset to a new CSV file
concatenated_data.to_csv('mnist_gaussian_hog_concatenated.csv', index=False)

print("Concatenated dataset has been saved to mnist_gaussian_hog_concatenated.csv")

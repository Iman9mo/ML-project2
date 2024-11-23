import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Read the MNIST dataset from CSV
mnist_data = pd.read_csv('mnist_sobel_hog_concatenated.csv')

# Extract images and labels
images = mnist_data.iloc[:, :-1].values  # All columns except the last
labels = mnist_data.iloc[:, -1].values    # Last column (labels)

# Step 2: Calculate the mean image and center the images
mean_image = np.mean(images, axis=0)
centered_images = images - mean_image  # Centering the images

# Step 3: Visualize some of the centered images
# def visualize_images(images, num_images=5):
#     plt.figure(figsize=(10, 5))
#     for i in range(num_images):
#         plt.subplot(1, num_images, i + 1)
#         plt.imshow(images[i].reshape(28, 28), cmap='gray')
#         plt.axis('off')
#     plt.show()
#
# # Visualize the mean image
# plt.figure(figsize=(5, 5))
# plt.imshow(mean_image.reshape(28, 28), cmap='gray')
# plt.title('Mean Image')
# plt.axis('off')
# plt.show()
#
# # Visualize some centered images
# visualize_images(centered_images, num_images=5)

# Step 4: Apply PCA to the centered images
pca = PCA()
pca.fit(centered_images)

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate cumulative variance
cumulative_variance = np.cumsum(explained_variance_ratio) * 100
n_component = np.argmax(cumulative_variance >= 60) + 1
print(n_component)
# Plotting
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Variance Explained')
plt.ylim(0, 110)
plt.axhline(y=60, color='r', linestyle='--')
plt.plot(cumulative_variance)
plt.show()


X_reduced = (PCA(n_components=40)).fit_transform(centered_images)
print(X_reduced.shape)
X_reduced_df = pd.DataFrame(X_reduced)

X_reduced_df['label'] = labels
X_reduced_df.to_csv('sobel_hog_reduced.csv', index=False)

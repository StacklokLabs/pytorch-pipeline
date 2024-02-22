import numpy as np
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Save the dataset to a file
np.savez('dataset.npz', X=X, y=y)

# Print information about the generated dataset
print("Synthetic dataset generated successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Class distribution:")
print("Class 0:", np.sum(y == 0))
print("Class 1:", np.sum(y == 1))

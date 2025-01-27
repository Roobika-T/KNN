import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class KNN:
    """
    A class to represent the K-Nearest Neighbors classifier.

    Attributes:
    -----------
    k : int
        Number of neighbors to consider.
    type_of_distance : str
        Type of distance metric to use ('euclidean' or 'manhattan').
    weightage_of_distance : str
        Weighting scheme for distances ('uniform' or 'distance').

    Methods:
    --------
    fit(X, y):
        Fits the model using the training data.
    calculate_distance(X):
        Computes the distance matrix based on the specified distance metric.
    predict(X):
        Predicts the labels for the input data X.
    _get_weighted_votes(k_nearest_labels, k_nearest_distances):
        Returns the most common label among the nearest neighbors, considering distance weightage.
    """

    def __init__(self, k=3, type_of_distance='euclidean', weightage_of_distance='uniform'):
        """
        Initializes the KNN classifier with the specified parameters.

        Parameters:
        -----------
        k : int
            Number of neighbors to consider.
        type_of_distance : str
            Type of distance metric to use ('euclidean' or 'manhattan').
        weightage_of_distance : str
            Weighting scheme for distances ('uniform' or 'distance').
        """
        if k <= 0:
            raise ValueError("Number of neighbors must be positive.")
        if type_of_distance not in ['euclidean', 'manhattan']:
            raise ValueError("type_of_distance must be 'euclidean' or 'manhattan'.")
        if weightage_of_distance not in ['uniform', 'distance']:
            raise ValueError("weightage_of_distance must be 'uniform' or 'distance'.")

        self.k = k
        self.type_of_distance = type_of_distance
        self.weightage_of_distance = weightage_of_distance

    def fit(self, X, y):
        """Fit the KNN model with training data."""
        self.X_train = X
        self.y_train = y

    def calculate_distance(self, X):
        """Calculate distances between the input data and the training data."""
        if self.type_of_distance == 'euclidean':
            distances = np.sqrt(((X[:, np.newaxis] - self.X_train) ** 2).sum(axis=2))
        elif self.type_of_distance == 'manhattan':
            distances = np.abs(X[:, np.newaxis] - self.X_train).sum(axis=2)
        return distances

    def _get_weighted_votes(self, k_nearest_labels, k_nearest_distances):
        """Return the most common label considering distance weightage."""
        if self.weightage_of_distance == 'uniform':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        elif self.weightage_of_distance == 'distance':
            weight_sum = {}
            for label, distance in zip(k_nearest_labels, k_nearest_distances):
                weight_sum[label] = weight_sum.get(label, 0) + 1 / distance if distance != 0 else 0
            return max(weight_sum, key=weight_sum.get)

    def predict(self, X):
        """Predict the class labels for the input data X."""
        distance_matrix = self.calculate_distance(X)
        predictions = []
        for distances in distance_matrix:
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            k_nearest_distances = distances[k_indices]
            label = self._get_weighted_votes(k_nearest_labels, k_nearest_distances)
            predictions.append(label)
        return np.array(predictions)

# Utility Functions
def standardize_data(X):
    """Standardize the dataset (zero mean, unit variance)."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def normalize_data(X):
    """Normalize the dataset (values between 0 and 1)."""
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def label_encode(y):
    """Encode categorical labels as integers."""
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([label_map[label] for label in y])

def plot_data(X, y, title="Data Visualization"):
    """Plot the data using PCA for dimensionality reduction."""
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

def calculate_accuracy(predictions, true_labels):
    """Calculate the accuracy of the model."""
    return np.mean(predictions == true_labels)

def main():
    try:
        # Load data with tab delimiter
        data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Implementing KNN Algorithm.csv', delimiter='\t')

        # Separate features and labels
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Verify feature shape
        print("Data shape:", data.shape)
        print("Features shape:", X.shape)
        print("Labels shape:", y.shape)

        if X.shape[1] == 0:
            raise ValueError("Feature matrix has zero features.")

        # Standardize the data
        X = standardize_data(X)

        # Label encode the target
        y = label_encode(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Plot the training data
        plot_data(X_train, y_train, title="Standardized Training Data")

        # Get user input for distance choice
        distance_choice = input("Choose the type of distance calculation ('euclidean' or 'manhattan'): ").strip().lower()

        # Validate the input
        if distance_choice not in ['euclidean', 'manhattan']:
            raise ValueError("Invalid choice. Please choose 'euclidean' or 'manhattan'.")

        # Initialize the KNN model with the user's choice
        knn = KNN(k=3, type_of_distance=distance_choice, weightage_of_distance='uniform')

        # Fit the model on the training set
        knn.fit(X_train, y_train)

        # Predict on the testing set
        predictions = knn.predict(X_test)

        # Print the predictions
        print("Predicted labels:", predictions)

        # Calculate accuracy
        accuracy = calculate_accuracy(predictions, y_test)
        print(f"Accuracy: {accuracy:.2f}")

        # Plot the test predictions
        plot_data(X_test, predictions, title="Predicted Labels on Test Set")

    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

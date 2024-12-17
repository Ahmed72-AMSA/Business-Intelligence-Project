import numpy as np
from scipy.stats import mode

class KNearestNeighborsClassifier:
    def __init__(self, K):
        self.K = K  

    def fit(self, X_train, y_train):
        # Store the training data
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        # Predict labels for the test set
        X_test = np.array(X_test)
        y_pred = np.zeros(X_test.shape[0], dtype=int)

        for i in range(X_test.shape[0]):
            test_instance = X_test[i]
            neighbors = self._find_neighbors(test_instance)
            
            mode_result = mode(neighbors)  # Most frequent class
            most_frequent_class = mode_result[0]
            y_pred[i] = most_frequent_class

        return y_pred

    def predict_proba(self, X_test):
        X_test = np.array(X_test)
        probas = np.zeros((X_test.shape[0], len(np.unique(self.y_train))))  # One column for each class
        
        for i in range(X_test.shape[0]):
            test_instance = X_test[i]
            neighbors = self._find_neighbors(test_instance)
            
            # Calculate the frequency of each class in the neighbors
            unique_classes, counts = np.unique(neighbors, return_counts=True)
            
            # Convert counts to probabilities by dividing by K (number of neighbors)
            probas[i, unique_classes] = counts / self.K

        return probas

    def _find_neighbors(self, test_instance):
        # Compute the Euclidean distance between test_instance and all training samples
        distances = np.linalg.norm(self.X_train - test_instance, axis=1)
        
        # Get the indices of the K nearest neighbors
        nearest_indices = np.argsort(distances)[:self.K]
        
        # Get the class labels of the K nearest neighbors
        neighbors = self.y_train[nearest_indices]
        return neighbors

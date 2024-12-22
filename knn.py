import numpy as np

def euclidean(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))

def hamming(f1, f2):
    return np.sum(f1 != f2, axis=1)

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train, dtype=float)
        self.y_train = np.array(y_train)

    def dist_metric(self, x, X_train): 
        return np.sqrt(np.sum((X_train - x) ** 2, axis=1))

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=float)
        predictions = []

        for x in X_test:
            distances = self.dist_metric(x, self.X_train)

            nearest_indices = np.argpartition(distances, self.k)[:self.k]
            nearest_distances = distances[nearest_indices]
            nearest_labels = self.y_train[nearest_indices]

            # Weighted voting: Use inverse of distance as weight
            weights = 1 / (nearest_distances + 1e-5)
            weighted_votes = {}
            for label, weight in zip(nearest_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight

            predictions.append(max(weighted_votes, key=weighted_votes.get))

        return predictions

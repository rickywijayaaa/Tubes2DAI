import numpy as np
from concurrent.futures import ThreadPoolExecutor

class KNN:
    def __init__(self, k=3, metric='euclidean', p=1):
        self.k = k
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)  # Use float32 for smaller memory
        self.y_train = np.array(y, dtype=np.int32)    # Integer labels for indexing

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'manhattan', or 'minkowski'.")

    def _get_neighbors(self, x):
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        neighbors_idx = np.argsort(distances)[:self.k]
        neighbors_distances = np.array(distances)[neighbors_idx]
        return neighbors_idx, neighbors_distances

    def _predict_single(self, x):
        neighbors_idx, neighbors_distances = self._get_neighbors(x)
        neighbor_labels = self.y_train[neighbors_idx]

        # Weighted voting
        weights = 1 / (neighbors_distances + 1e-5)  # Avoid division by zero
        weighted_votes = {}
        for label, weight in zip(neighbor_labels, weights):
            weighted_votes[label] = weighted_votes.get(label, 0) + weight

        # Return the class with the highest weighted vote
        return max(weighted_votes, key=weighted_votes.get)

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float32)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def predict_parallel(self, X_test, num_workers=4):
        """Parallelize predictions across multiple CPU cores."""
        X_test_split = np.array_split(X_test, num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.predict, X_test_split))
        return np.concatenate(results)

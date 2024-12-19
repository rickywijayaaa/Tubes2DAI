import numpy as np

class KNN:
    def __init__(self, k=3, metric='euclidean', p=1):
        self.k = k
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y, dtype=int)  # Ensure y_train is integers

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        elif self.metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm_x1 = np.sqrt(np.sum(x1 ** 2))
            norm_x2 = np.sqrt(np.sum(x2 ** 2))
            return 1 - (dot_product / (norm_x1 * norm_x2))
        elif self.metric == 'chebyshev':
            return np.max(np.abs(x1 - x2))
        elif self.metric == 'hamming':
            return np.sum(x1 != x2) / len(x1)
        else:
            raise ValueError("Invalid metric. Choose 'euclidean', 'manhattan', 'minkowski', 'cosine', 'chebyshev', or 'hamming'.")

    def _get_neighbors(self, x):
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        neighbors_idx = np.argsort(distances)[:self.k]
        return neighbors_idx

    def predict(self, X):
        predictions = []
        for x in np.array(X):
            neighbors_idx = self._get_neighbors(x)
            neighbor_labels = self.y_train[neighbors_idx]
            predicted_label = np.bincount(neighbor_labels).argmax()
            predictions.append(predicted_label)
        return np.array(predictions)

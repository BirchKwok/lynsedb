import numpy as np
import torch
from spinesTS.utils import seed_everything

from min_vec.engine import get_device


class KMeans:
    def __init__(self, n_clusters, distance='cosine', random_state=42, epochs=100, learning_rate=0.1,
                 batch_size=1000, device='auto'):
        seed_everything(random_state)

        self.device = get_device(device)
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.centroids = None
        self._labels = None
        self.distance = distance

        self.distance_func = self.cosine_distance if distance == 'cosine' else self.euclidean_distance

    @property
    def cluster_centers_(self):
        return self.centroids.detach().cpu().numpy()

    @property
    def labels_(self):
        return self._labels

    @staticmethod
    def cosine_distance(a, b):
        b = b.unsqueeze(0)
        return 1 - torch.nn.functional.cosine_similarity(a.unsqueeze(1), b, dim=2)

    @staticmethod
    def euclidean_distance(a, b):
        b = b.unsqueeze(0)
        return torch.norm(a.unsqueeze(1) - b, dim=2)

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(self.device)
        indices = torch.randperm(X.shape[0])[:self.n_clusters]
        self.centroids = X[indices]

        for _ in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                self._update_centroids(X_batch)

        self._labels = self.predict(X)

        return self

    def _update_centroids(self, X):
        distances = self.distance_func(X, self.centroids)
        labels = torch.argmin(distances, dim=1)

        for i in range(self.n_clusters):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                new_centroid = points_in_cluster.mean(0)
                self.centroids[i] = self.learning_rate * new_centroid + (1 - self.learning_rate) * self.centroids[i]

    def partial_fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        X = X.to(self.device)

        if self.centroids is None:
            self.centroids = X[:self.n_clusters]
        else:
            self._update_centroids(X)

        self._labels = self.predict(X)

        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(self.device)
        distances = self.distance_func(X, self.centroids)
        labels = torch.argmin(distances, dim=1)
        return labels.detach().cpu().numpy()

    def save(self, filename):
        torch.save(self.centroids.detach().cpu(), filename)

    def load(self, filename):
        self.centroids = torch.load(filename).to(self.device)

        return self

import portalocker
from sklearn.cluster import MiniBatchKMeans


class BatchKMeans:
    def __init__(self, n_clusters, random_state=42, epochs=100, batch_size=1024):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, max_iter=epochs,
                                     batch_size=batch_size, n_init='auto', reassignment_ratio=0)

        self.n_clusters = n_clusters
        self.epochs = epochs
        self.batch_size = batch_size

        self.fitted = False

    def fit(self, X):
        self.model.fit(X)
        self.fitted = True

        return self

    def partial_fit(self, X):
        self.model.partial_fit(X)
        self.fitted = True

        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError('The model is not fitted yet. Please call the fit method first.')

        return self.model.predict(X)

    def save(self, filename):
        import cloudpickle

        with open(filename, 'wb') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            cloudpickle.dump(self.model, f)

    def load(self, filename):
        import cloudpickle

        self.model = cloudpickle.load(open(filename, 'rb'))

        return self

    @property
    def cluster_centers_(self):
        if not self.fitted:
            raise ValueError('The model is not fitted yet. Please call the fit method first.')
        return self.model.cluster_centers_

    @property
    def labels_(self):
        if not self.fitted:
            raise ValueError('The model is not fitted yet. Please call the fit method first.')
        return self.model.labels_

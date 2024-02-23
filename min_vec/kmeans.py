import numpy as np
from sklearn.cluster import MiniBatchKMeans


# class MiniBatchKMeansWithCosine(MiniBatchKMeans):
#     def __init__(
#         self,
#         n_clusters=8,
#         *,
#         init="k-means++",
#         max_iter=100,
#         batch_size=1024,
#         verbose=0,
#         compute_labels=True,
#         random_state=None,
#         tol=0.0,
#         max_no_improvement=10,
#         init_size=None,
#         n_init="warn",
#         reassignment_ratio=0.01,
#     ):
#         super().__init__(
#             n_clusters=n_clusters,
#             init=init,
#             max_iter=max_iter,
#             verbose=verbose,
#             random_state=random_state,
#             tol=tol,
#             n_init=n_init,
#         )
#
#         self.max_no_improvement = max_no_improvement
#         self.batch_size = batch_size
#         self.compute_labels = compute_labels
#         self.init_size = init_size
#         self.reassignment_ratio = reassignment_ratio
#
#
#     def fit(self, X, y=None, weight=None):
#         X_normalized = X / np.linalg.norm(X, axis=1)[:, np.newaxis]  # 计算数据的单位向量
#         return super().fit(X_normalized, y)
#
#     def _init_batch(self, X, x_squared_norms, init_size):
#         self.batch_center_ids_ = self._rng.integers(
#             0, X.shape[0], size=self.n_clusters
#         )
#         self.cluster_centers_ = X[self.batch_center_ids_]
#
#     def _update_batch(self, X, x_squared_norms):
#         centers_squared_norms = np.sum(self.cluster_centers_ ** 2, axis=1)
#
#         distances = np.dot(X, self.cluster_centers_.T)
#         distances *= -2
#         distances += x_squared_norms[:, np.newaxis]
#         distances += centers_squared_norms
#
#         best_centers = np.argmin(distances, axis=1)
#
#         # 将相似度转换为距离
#         # 根据相似度的计算方式，距离越小，相似度越大，所以需要取反
#         self.cluster_centers_ += (
#             self.learning_rate
#             / (1 + self.n_iter_ * self.batch_size)
#             ) * (
#             X[best_centers] - self.cluster_centers_
#         )
#
#         self._mini_batch_step(X, x_squared_norms, best_centers)


class KMeans:
    def __init__(self, n_clusters, random_state=42, epochs=100, batch_size=1024, distance='euclidean'):
        # if distance == 'euclidean':
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, max_iter=epochs,
                                     batch_size=batch_size, n_init='auto')
        # else:
        #     self.model = MiniBatchKMeansWithCosine(n_clusters=n_clusters, random_state=random_state, max_iter=epochs,
        #                                            batch_size=batch_size, n_init='auto')

        self.n_clusters = n_clusters
        self.distance = distance
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

        cloudpickle.dump(self.model, open(filename, 'wb'))

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

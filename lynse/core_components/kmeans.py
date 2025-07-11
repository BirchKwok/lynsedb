from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import wasserstein_distance
import logging


class EnhancedBatchKMeans:
    def __init__(self, n_clusters, random_state=42, epochs=100, batch_size=1024,
                 drift_threshold=0.3, quality_threshold=0.5):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                                   max_iter=epochs, batch_size=batch_size,
                                   n_init='auto', reassignment_ratio=0)

        self.n_clusters = n_clusters
        self.epochs = epochs
        self.batch_size = batch_size
        self.drift_threshold = drift_threshold
        self.quality_threshold = quality_threshold

        self.fitted = False
        self.reference_distribution = None
        self.quality_scores_history = []

    def _compute_quality_score(self, X):
        """计算聚类质量分数"""
        if len(X) < self.n_clusters * 2:  # 样本太少，无法计算有效的质量分数
            return 1.0

        try:
            labels = self.predict(X)
            silhouette = silhouette_score(X, labels, sample_size=min(10000, len(X)))
            calinski = calinski_harabasz_score(X, labels)
            # 综合评分 (归一化后的加权平均)
            return (0.6 * (silhouette + 1) / 2 + 0.4 * min(calinski / 10000, 1))
        except Exception as e:
            logging.warning(f"Failed to compute quality score: {e}")
            return 1.0

    def _detect_distribution_drift(self, X):
        """检测数据分布是否发生显著漂移"""
        if self.reference_distribution is None:
            return False

        # 计算新数据到聚类中心的距离分布
        new_distances = self.model.transform(X).min(axis=1)

        # 使用 Wasserstein 距离比较分布变化
        drift_score = wasserstein_distance(new_distances, self.reference_distribution)

        return drift_score > self.drift_threshold

    def _update_reference_distribution(self, X):
        """更新参考分布"""
        distances = self.model.transform(X).min(axis=1)
        self.reference_distribution = distances

    def fit(self, X: np.ndarray):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.model.fit(X)
        self.fitted = True
        self._update_reference_distribution(X)

        quality_score = self._compute_quality_score(X)
        self.quality_scores_history.append(quality_score)

        return self

    def partial_fit(self, X: np.ndarray):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        # 检查是否需要重新训练
        if self.fitted:
            drift_detected = self._detect_distribution_drift(X)
            current_quality = self._compute_quality_score(X)

            if drift_detected or current_quality < self.quality_threshold:
                logging.info("Significant drift detected or quality degraded. Reinitializing model...")
                self.model = MiniBatchKMeans(n_clusters=self.n_clusters,
                                           random_state=42,
                                           max_iter=self.epochs,
                                           batch_size=self.batch_size,
                                           n_init='auto')

        self.model.partial_fit(X)
        self.fitted = True
        self._update_reference_distribution(X)

        quality_score = self._compute_quality_score(X)
        self.quality_scores_history.append(quality_score)

        return self

    def predict(self, X: np.ndarray):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if not self.fitted:
            raise ValueError('The model is not fitted yet. Please call the fit method first.')

        return self.model.predict(X)

    def get_quality_trend(self):
        """返回聚类质量趋势"""
        return self.quality_scores_history

    def save(self, filename):
        import cloudpickle

        with open(filename, 'wb') as f:
            state = {
                'model': self.model,
                'n_clusters': self.n_clusters,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'reference_distribution': self.reference_distribution,
                'quality_scores_history': self.quality_scores_history,
                'drift_threshold': self.drift_threshold,
                'quality_threshold': self.quality_threshold
            }
            cloudpickle.dump(state, f)

    def load(self, filename):
        import cloudpickle

        with open(filename, 'rb') as f:
            state = cloudpickle.load(f)
            self.model = state['model']
            self.n_clusters = state['n_clusters']
            self.epochs = state['epochs']
            self.batch_size = state['batch_size']
            self.reference_distribution = state['reference_distribution']
            self.quality_scores_history = state['quality_scores_history']
            self.drift_threshold = state['drift_threshold']
            self.quality_threshold = state['quality_threshold']
            self.fitted = True

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

# 保持原有的 BatchKMeans 类作为后备和兼容性支持
class BatchKMeans:
    def __init__(self, n_clusters, random_state=42, epochs=100, batch_size=1024):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, max_iter=epochs,
                                     batch_size=batch_size, n_init='auto', reassignment_ratio=0)

        self.n_clusters = n_clusters
        self.epochs = epochs
        self.batch_size = batch_size

        self.fitted = False

    def fit(self, X: np.ndarray):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.model.fit(X)
        self.fitted = True

        return self

    def partial_fit(self, X: np.ndarray):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.model.partial_fit(X)
        self.fitted = True

        return self

    def predict(self, X: np.ndarray):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if not self.fitted:
            raise ValueError('The model is not fitted yet. Please call the fit method first.')

        return self.model.predict(X)

    def save(self, filename):
        import cloudpickle

        with open(filename, 'wb') as f:
            cloudpickle.dump([self.model, self.n_clusters, self.epochs, self.batch_size], f)

    def load(self, filename):
        import cloudpickle

        self.model, self.n_clusters, self.epochs, self.batch_size = cloudpickle.load(open(filename, 'rb'))
        self.fitted = True

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

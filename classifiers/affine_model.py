import numpy as np
from sklearn.decomposition import PCA
from exceptions import Exception

from time import time

class AffineModel:
    '''Classification algorithm described in BrunaMallat 2011'''

    def __init__(self, n_components):
        self.n_components = n_components
        self.models = []

    def fit(self, data, labels):
        for l in range(0, labels.max() + 1):
            d = data[labels == l]
            pca = PCA(n_components=self.n_components)
            pca.fit(d)
            self.models.append(pca)

    def predict(self, data):
        if len(self.models) == 0:
            raise Exception("Can not be called before fit")
        t = time()
        res = map(self.classify, data)
        dt = time()-t
        # print "Prediction took\n\t%.3f s\n\t%.3f ms per point" % (dt, 1000*dt / len(data))
        return res

    def classify(self, data):
        # distances = [self.bruna_mallat_dist(m, data) for m in self.models]
        distances = [self.model_dist(m, data) for m in self.models]
        # distances = [self.mahalanobis_dist(m, data) for m in self.models]

        return np.argmin(distances)

    def mahalanobis_dist(self, model, data):
        assert (isinstance(model, PCA))
        transformed = model.transform(data).ravel()
        sd = np.sqrt(model.explained_variance_ratio_)
        dist = transformed / sd
        # print dist.shape
        return np.sum(np.power(dist, 2))

    def model_dist(self, model, data):
        assert (isinstance(model, PCA))
        transformed = model.transform(data)
        projection = model.inverse_transform(transformed)
        distance = np.sum(np.power(projection - data, 2))
        return distance

    def bruna_mallat_dist(self, model, data):
        assert (isinstance(model, PCA))

        s = data - model.mean_.ravel()

        transformed = model.transform(data)
        projection = model.inverse_transform(transformed)
        p = projection - model.mean_.ravel()

        return np.sum(np.power(s, 2)) - np.sum(np.power(p, 2))

    def _create_model(self, data):
        pca = PCA(n_components=self.n_components)
        pca.fit(data)

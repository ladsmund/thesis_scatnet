import numpy as np
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix


class AffineModelOld():
    '''Classification algorithm described in BrunaMallat 2011'''

    def __init__(self, **kwargs):
        self.n_components = kwargs.pop('n_components', None)
        self.models = []

    def get_params(self, deep=True):
        return {'n_components': self.n_components}

    def fit(self, data, labels):
        self.models = []
        self.n_components = None
        for l in range(0, labels.max() + 1):
            d = data[labels == l]

            pca = PCA(n_components=self.n_components)
            pca.fit(d)
            if self.n_components is None:
                self.n_components = pca.n_components_
            self.n_components = min(self.n_components, pca.n_components_)
            self.n_components = min(self.n_components, pca.n_samples_)
            self.models.append((pca.mean_, pca.components_.T, pca.explained_variance_ratio_))

        self.models = [(mean, np.mat(comp[:, :self.n_components]), var) for mean, comp, var in self.models]
        return self

    def classify(self, data, dim=None):
        # distances = [self.model_dist(m, data, dim=dim) for m in self.models]

        distances = np.zeros(shape=(data.shape[0], len(self.models)), dtype='float32')
        for i, m in enumerate(self.models):
            d = self.model_dist(m, data, dim=dim)
            distances[:, i] = np.ravel(d)


        # print distances
        # distances = [self.mahalanobis_dist(m, data) for m in self.models]
        return np.argmin(distances, axis=1)
        return np.argmin(distances, axis=0)

    def score(self, test_data, test_labels, dim=None, confussion_mat=False):
        labels = np.ravel(self.classify(test_data, dim))
        if not confussion_mat:
            count = np.sum(labels == test_labels)
            return float(count) / test_data.shape[0]
        else:
            return confusion_matrix(test_labels, labels)

    def mahalanobis_dist(self, model, data):
        transformed = model.transform(data).ravel()
        sd = np.sqrt(model.explained_variance_ratio_)
        dist = transformed / sd
        return np.sum(np.power(dist, 2))

    def model_dist(self, model, data, dim=None):
        if dim is None:
            dim = self.n_components - 1
        # print "dim: %i" % dim

        data_t = data - model[0]
        data_p = (data_t * model[1][:, :dim]) * model[1][:, :dim].T
        return np.sum(np.power(data_t - data_p, 2), axis=1)

        # transformed = (data - model[0]) * model[1][:, dim:]
        # distance = np.sum(np.power(transformed, 2), axis=1)
        # return distance

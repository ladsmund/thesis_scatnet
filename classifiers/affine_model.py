import os

import numpy as np
from sklearn.decomposition import PCA

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

import pickle


def reshape_to_1d(data):
    dim = np.product(data.shape[1:])
    return np.reshape(data, (data.shape[0], dim))

def mahalanobis_dist(model, data):
    transformed = model.transform(data).ravel()
    sd = np.sqrt(model.explained_variance_ratio_)
    dist = transformed / sd
    return np.sum(np.power(dist, 2))

def model_dist(model, data, dim=None):
    if dim is None:
        dim = data.shape[1]

    data_t = data - model[0]
    data_p = (data_t * model[1][:, :dim]) * model[1][:, :dim].T
    return np.sum(np.power(data_t - data_p, 2), axis=1)

def distance(model_path, *argv, **kwargs):
    model = np.load(model_path)
    return np.ravel(model_dist(model, *argv, **kwargs))


class AffineModel():
    def __init__(self, **kwargs):
        self.key = kwargs.pop('key', 'affine_mod')
        self.root_path = kwargs.pop('root_path', '/tmp/')
        self.n_components = kwargs.pop('n_components', None)
        self.model_keys = []
        self.models = []

    def get_params(self, deep=True):
        return {'n_components': self.n_components}

    def get_model_path(self, model_key):
        return os.path.join(self.root_path, model_key) + '.model'

    def train_classifier(self, label, data):
        model_key = "%s_%i" % (self.key, label)
        model_path = self.get_model_path(model_key)
        pca = PCA(n_components=self.n_components)
        pca.fit(data)
        if self.n_components is None:
            self.n_components = pca.n_components_
        self.n_components = min(self.n_components, pca.n_components_)
        self.n_components = min(self.n_components, pca.n_samples_)
        np.save(open(model_path, 'w'),
                (pca.mean_, np.mat(pca.components_.T, dtype=pca.components_.dtype), pca.explained_variance_ratio_))
        if not model_key in self.model_keys:
            self.model_keys.append(model_key)

    def fit(self, data, labels):
        self.models = []
        for l in set(labels):
            self.train_classifier(l, data[labels == l])
        return self

    def classify(self, data, dim=None):
        distances = np.zeros(shape=(data.shape[0], len(self.model_keys)), dtype='float32')
        for i, model_key in enumerate(self.model_keys):
            model_path = self.get_model_path(model_key)
            distances[:, i] = distance(model_path, data, dim=dim)
        # print distances
        return np.argmin(distances, axis=1)

    def score(self, test_data, test_labels, dim=None, confussion_mat=False):
        labels = np.ravel(self.classify(test_data, dim))
        if not confussion_mat:
            count = np.sum(labels == test_labels)
            return float(count) / test_data.shape[0]
        else:
            return confusion_matrix(test_labels, labels)

    def delete(self):
        for mk in self.model_keys:
            path = self.get_model_path(mk)
            os.remove(path)


def find_best_dimension(data, labels, ):
    folds = 4

    skf = cross_validation.StratifiedKFold(np.ravel(labels), folds)

    classifier_number = 0
    all_scores = {}
    for train, test in skf:

        classifier_number += 1
        print "classifier %i/%i" % (classifier_number, folds)
        # classifier = AffineModel()
        classifier = AffineModel()

        classifier.fit(data[train], labels[train])
        max_dim = classifier.n_components

        step = max(max_dim // 10, 1)
        search_dim = range(1, max_dim, step)

        classifier_scores = {}
        while len(search_dim) > 0:
            for d in search_dim:
                if d in classifier_scores:
                    continue
                classifier_scores[d] = classifier.score(data[test], labels[test], dim=d)
                print "  dim %5i: %6.3f%%" % (d, 100 * classifier_scores[d])
            best_dim = max(classifier_scores.items(), key=lambda i: i[1])[0]
            if step == 1:
                break
            new_step = max(int(step // 1.5), 1)
            search_dim = range(max(best_dim - step, 0), best_dim + step, new_step)
            step = new_step

        for d, score in classifier_scores.items():
            if d in all_scores:
                all_scores[d].append(score)
            else:
                all_scores[d] = [score]

        classifier.delete()

    scores = {}
    for d, score_list in all_scores.items():
        scores[d] = np.median(score_list)

    best_dim = max(scores.items(), key=lambda i: i[1])[0]

    # print best_dim
    # import matplotlib.pyplot as plt
    # dim, score = zip(*scores.items())
    # plt.plot(dim, score,'.')
    # plt.show()
    return best_dim

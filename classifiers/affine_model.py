import numpy as np
from sklearn.decomposition import PCA

from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


def reshape_to_1d(data):
    dim = np.product(data.shape[1:])
    return np.reshape(data, (data.shape[0], dim))


class AffineModel():
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
        distances = [self.model_dist(m, data, dim=dim) for m in self.models]
        # distances = [self.mahalanobis_dist(m, data) for m in self.models]
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
            dim = self.n_components-1
        # print "dim: %i" % dim
        transformed = (data - model[0]) * model[1][:, dim:]
        distance = np.sum(np.power(transformed, 2), axis=1)
        return distance


def find_best_dimension(data, labels):
    folds = 4

    skf = cross_validation.StratifiedKFold(np.ravel(labels), folds)

    classifier_number = 0
    all_scores = {}
    for train, test in skf:

        classifier_number += 1
        print "classifier %i/%i" % (classifier_number, folds)
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
                print "  dim %5i: %.2f" % (d, classifier_scores[d])
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

    scores = {}
    for d, score_list in all_scores.items():
        scores[d] = np.median(score_list)

    best_dim = max(scores.items(), key=lambda i: i[1])[0]

    print best_dim
    import matplotlib.pyplot as plt
    dim, score = zip(*scores.items())
    plt.plot(dim, score,'.')
    plt.show()
    return best_dim

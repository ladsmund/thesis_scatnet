import numpy as np
import pickle
import gzip

DEFAULT_RANDOM_SEED = 0

class Dataset:
    def __init__(self, **kwargs):
        if 'train_data' in kwargs:
            train_data = kwargs['train_data']
            train_labels = kwargs['train_labels']
            test_data = kwargs['test_data']
            test_labels = kwargs['test_labels']

            self.mask_test = np.concatenate([0*train_labels, 0*test_labels+ 1]).astype('bool')
            self.mask_train = (1 - self.mask_test).astype('bool')
            self.data = np.concatenate([train_data, test_data])
            self.labels = np.concatenate([train_labels, test_labels])
        else:
            self.mask_test = kwargs['mask_test']
            self.mask_train = kwargs['mask_train']
            self.data = kwargs['data']
            self.labels = kwargs['labels']

        self.label_set = set(self.labels)
        self.nclasses = len(self.label_set)

        self.parent_dataset = []
        if 'parent_dataset' in kwargs:
            self.parent_dataset = kwargs['parent_dataset']

    def seed(self, seed=DEFAULT_RANDOM_SEED):
        np.random.seed(seed)

    def get_subset(self, count_train, count_test):
        train_data, train_labels = self.get_data(test=False)
        test_data, test_labels = self.get_data(test=True)

        train_data, train_labels = self._select_subset(train_data, train_labels, count_train)
        test_data, test_labels = self._select_subset(test_data, test_labels, count_test)

        subdataset = Dataset(train_data=train_data,
                             train_labels=train_labels,
                             test_data=test_data,
                             test_labels=test_labels,
                             parent_dataset=self.parent_dataset+[self.__class__])
        return subdataset

    def _select_subset(self, data, labels, count):
        count_per_label = float(count) / self.nclasses

        out_data = []
        out_labels = []
        residual = 0
        for label in self.label_set:
            label_count = int(round(count_per_label + residual))
            residual += count_per_label - label_count

            l = (labels[labels == label])[:label_count]
            out_data.append((data[labels == label])[:label_count])
            out_labels.append((labels[labels == label])[:label_count])

        out_data = np.concatenate(out_data)
        out_labels = np.concatenate(out_labels)

        indx = np.random.permutation(np.arange(count))
        return out_data[indx], out_labels[indx]

    def get_data(self, count=None, test=False):
        if test:
            data, labels = self.data[self.mask_test], self.labels[self.mask_test]
        else:
            data, labels = self.data[self.mask_train], self.labels[self.mask_train]

        if count >= len(labels):
            count = None
        if count is not None:
            return self._select_subset(data, labels, count)
        else:
            return data, labels


def save_dataset(dataset, path):
    f = gzip.open(path, 'wb')
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def load_dataset(path):
    f = gzip.open(path, 'rb')
    return pickle.load(f)
    f.close()


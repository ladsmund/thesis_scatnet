import numpy as np
import tempfile
import sys

def reshape_to_1d(data):
    dim = np.product(data.shape[1:])

    # print "reshape: %s" % str(data.reshape((data.shape[0], dim)).shape)
    return data.reshape((data.shape[0], dim))

def normalize_coeff(data, **kwargs):
    reducer = kwargs.pop('reducer', 'max')

    norm_axis = tuple(range(2, len(data.shape)))
    b = np.sqrt(np.sum(np.power(data, 2), axis=norm_axis))

    c = None
    if reducer == 'max':
        c = np.max(b, axis=0)
    elif reducer == 'var':
        c = np.var(b, axis=0)
    else:
        sys.stderr('Uknown reduction function: %s' % reducer)
    return c


def normalize(data, **kwargs):
    output = kwargs.pop('output', None)
    if output is None:
        output_file = kwargs.pop('output_file', tempfile.TemporaryFile())
        output = np.memmap(output_file, shape=data.shape, dtype=data.dtype)
    output[:] = data[:]
    # output = data.copy()
    coeff = kwargs.pop('coefficients', normalize_coeff(data, **kwargs))

    for i, c in enumerate(coeff):
        if c == 0:
            print 'c == 0'
            continue
        output[:, i] /= c
    return output


class Normalizer:
    def __init__(self, reshape_to_1d=False):
        self.coeff = None
        self.reshape_to_1d = reshape_to_1d

    def fit(self, data, **kwargs):
        self.coeff = normalize_coeff(data, **kwargs)
        return self.normalize(data, **kwargs)

    def normalize(self, data, **kwargs):
        kwargs['coefficients'] = self.coeff
        if self.reshape_to_1d:
            return reshape_to_1d(normalize(data, **kwargs))
        else:
            return normalize(data, **kwargs)


def fit(classifier, data, *args, **kwargs):
    data_n = classifier.normalizer.fit(data)
    return classifier.__class__.fit(classifier, data_n)


def classify(classifier, data, *args, **kwargs):
    data_n = classifier.normalizer.normalize(data)
    return classifier.__class__.classify(classifier, data_n)


def extend_classifier(c, **kwargs):
    c_class = c.__class__
    setattr(c, 'normalizer', Normalizer(**kwargs))
    setattr(c, 'fit', lambda d, *args, **kwargs: c_class.fit(c, c.normalizer.fit(d), *args, **kwargs))
    setattr(c, 'classify', lambda d, *args, **kwargs: c_class.classify(c, c.normalizer.normalize(d), *args, **kwargs))
    return c


if __name__ == '__main__':
    data = np.random.random((10000, 20, 7, 7))
    data_n = normalize(data)

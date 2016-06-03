import numpy as np
import tempfile
import sys

from memory_profiler import profile


@profile
def normalize_coeff(input, reducer='max'):
    print input.shape[2:]
    norm_axis = tuple(range(2, len(input.shape)))
    b = np.sqrt(np.sum(np.power(data, 2), axis=norm_axis))

    c = None
    if reducer == 'max':
        c = np.max(b, axis=0)
    elif reducer == 'var':
        c = np.var(b, axis=0)
    else:
        sys.stderr('Uknown reduction function: %s' % reducer)
    return c

@profile
def normalize(data, **kwargs):
    output = kwargs.pop('output', None)
    if output is None:
        output_file = kwargs.pop('output_file', tempfile.TemporaryFile())
        output = np.memmap(output_file, shape=data.shape, dtype=data.dtype)

    output[:] = data[:]
    coeff = normalize_coeff(data, **kwargs)
    for i, c in enumerate(coeff):
        if c == 0:
            print 'c == 0'
            continue
        output[:, i] /= c
    return output

if __name__ == '__main__':
    data = np.random.random((10000, 20, 7, 7))
    data_n = normalize(data)

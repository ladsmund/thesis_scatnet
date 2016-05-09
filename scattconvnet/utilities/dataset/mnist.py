import os
import urllib
import cPickle
import gzip

import numpy as np
import theano
import theano.tensor as T

import data_folder
import dataset

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
file_name = "mnist.pkl.gz"
file_path = os.path.join(data_folder.get_path(), file_name)

if not os.path.exists(file_path):
    print "mcl_util.dataset.mnist: downloading MNIST data file"
    mnist_file = urllib.URLopener()
    mnist_file.retrieve(MNIST_URL, file_path)


def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_digits():
    # Load the dataset
    f = gzip.open(file_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set


class MNIST(dataset.Dataset):
    def __init__(self):
        train_set, valid_set, test_set = load_digits()
        test_data, test_labels = test_set
        valid_data, valid_labels = valid_set
        train_data, train_labels = train_set

        train_data = np.concatenate([train_data, valid_data])
        train_labels = np.concatenate([train_labels, valid_labels])


        train_data = np.reshape(train_data, (len(train_data),28,28))
        test_data = np.reshape(test_data, (len(test_data),28,28))

        dataset.Dataset.__init__(self, train_data, train_labels, test_data, test_labels)
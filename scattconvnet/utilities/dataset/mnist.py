import os
import urllib
import cPickle
import gzip

import numpy as np

import data_folder
import dataset

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
file_name = "mnist.pkl.gz"
file_path = os.path.join(data_folder.get_path(), file_name)

if not os.path.exists(file_path):
    print "mcl_util.dataset.mnist: downloading MNIST data file"
    mnist_file = urllib.URLopener()
    mnist_file.retrieve(MNIST_URL, file_path)

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

        dataset.Dataset.__init__(self,
                                 train_data=train_data,
                                 train_labels=train_labels,
                                 test_data=test_data,
                                 test_labels=test_labels)
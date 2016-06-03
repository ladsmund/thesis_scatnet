import argparse
import os
import numpy as np
from utilities.dataset import Dataset
import scattconvnet
from classifiers.affine_model import AffineModel, reshape_to_1d, find_best_dimension
import normalizer
import conduct_exp

from time import time

parser = argparse.ArgumentParser()
parser.add_argument('mnist_path')
parser.add_argument('--split', required=True, type=int)
parser.add_argument('--source_key', type=str, default='original')
parser.add_argument('--labels_key', type=str, default='labels')
parser.add_argument('--redo_scatnet', type=bool, default=False)
parser.add_argument('--redo_affine_model', type=bool, default=False)
parser.add_argument('--redo_find_dim', type=bool, default=False)
parser.add_argument('-a', '--nangles', type=int, default=6)
parser.add_argument('-s', '--scale', type=int, default=3)
parser.add_argument('-m', '--max_depth', type=int, default=3)
args = parser.parse_args()
mnist_path = args.mnist_path
train_test_split = args.split
source_key = args.source_key
labels_key = args.labels_key
redo_scatnet = args.redo_scatnet
redo_affine_model = args.redo_affine_model
redo_find_dim = args.redo_find_dim
parameters = {}
parameters["nangles"] = args.nangles
parameters["scale"] = args.scale
parameters["max_depth"] = args.max_depth

print("****************************")
print("Load Dataset")
t0 = time()
t = time()

dataset = Dataset(mnist_path)

labels = dataset.get_asset_data(labels_key)
data_key = "scatnet_" + "_".join(["%s%i" % (i) for i in parameters.items()]) + "_" + source_key
data_normalized_key = "normalized"
affine_model_key = 'affine_model_' + data_key
best_dim_key = 'best_dim_' + data_key
print "data_key:         %s" % data_key
print "affine_model_key: %s" % affine_model_key

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("****************************")
print("Process ScatNet Coefficients")
if redo_scatnet or not data_key in dataset.assets:
    data = scattconvnet.process_data(key=data_key, dataset=dataset, input_asset_key=source_key, **parameters)
    dataset.save()
else:
    data = dataset.get_asset_data(data_key)

# data = reshape_to_1d(data)

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("*******************************")
print("Select test and train data sets")
t = time()

print train_test_split
print data.shape

data_train = data[:train_test_split]
data_test = data[train_test_split:]

labels_train = labels[:train_test_split]
labels_test = labels[train_test_split:]

dt = time() - t

print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("************************************")
print("Generate affine model for all labels")
t = time()


classifier = AffineModel(root_path=mnist_path, key=affine_model_key)
# data_train = np.mat(reshape_to_1d(data_train))
# data_test = np.mat(reshape_to_1d(data_test))
classifier = normalizer.extend_classifier(classifier, reshape_to_1d=True)



dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("*******************")
print("Test classification")

t = time()
conduct_exp.conduct_experiment(
    classifier=classifier,
    n_training_samples=[100, 300, 500, 600],
    training_data=data_train,
    training_labels=labels_train,
    test_data=data_test,
    test_labels=labels_test
)

classifier.fit(data_train, labels_train, find_dim=True)
for _ in range(10):
    score = classifier.score(data_test, labels_test)
    dataset.assets[affine_model_key]['score'] = "%.3f%%" % (100 * score)
    dataset.save()
print "Accuricy: %.3f%%" % (100 * score)

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt / 10)

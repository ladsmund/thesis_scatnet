import argparse
import os
import numpy as np
from utilities.dataset import Dataset
import scattconvnet
from classifiers.affine_model import AffineModel, reshape_to_1d, find_best_dimension

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
print("*******************")
print("Batch normalization")
t = time()


data_normalized_path = os.path.join(mnist_path, data_normalized_key + ".data")
if not data_normalized_key in dataset.assets[data_key]:

    data_norm = np.linalg.norm(data_train, axis=(2, 3))
    max_norm = np.max(data_norm, axis=0)

    for i, m in enumerate(max_norm):
        if m == 0 or m == 1:
            continue
        data[:, i, :, :] /= m

    dataset.assets[data_key][data_normalized_key] = 'batch_max'
    dataset.save()

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)

data_train = np.mat(reshape_to_1d(data_train))
data_test = np.mat(reshape_to_1d(data_test))

print("******************************************")
print("Determine dimensionality for affine models")
t = time()
best_dim = 0
if redo_find_dim or best_dim_key not in dataset.assets:
    best_dim = find_best_dimension(data_train, labels_train)
    dataset.assets[best_dim_key] = best_dim
else:
    best_dim = int(dataset.assets[best_dim_key])
print "best_dim: %i" % best_dim

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("************************************")
print("Generate affine model for all labels")
t = time()
affine_model = None
if True or redo_affine_model or not affine_model_key in dataset.assets:
    affine_model = AffineModel(n_components=None, verbose=True, root_path=mnist_path, key=affine_model_key)
    affine_model.fit((data_train), labels_train)
    dataset.add_pickle_asset(affine_model, key=affine_model_key, parent_asset=data_key)
    dataset.save()

else:
    affine_model = dataset.get_asset_data(affine_model_key)

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("*******************")
print("Test classification")
t = time()
for _ in range(10):
    score = affine_model.score(data_test, labels_test, dim=best_dim)
    dataset.assets[affine_model_key]['score'] = "%.3f%%" % (100*score)
    dataset.save()
print "Accuricy: %.3f%%" % (100 * score)

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt / 10)

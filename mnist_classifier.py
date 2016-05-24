import argparse
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
affine_model_key = 'affine_model_' + data_key
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
data = reshape_to_1d(data)


dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("*******************************")
print("Select test and train data sets")
t = time()
data_train = data[:train_test_split]
data_test = data[train_test_split:]
labels_train = labels[:train_test_split]
labels_test = labels[train_test_split:]


dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("************************************")
print("Generate affine model for all labels")
t = time()
max_dim = 40
if redo_affine_model or not affine_model_key in dataset.assets:
    affine_model = AffineModel(n_components=max_dim, verbose=True)
    affine_model.fit(data_train, labels_train)
    dataset.add_pickle_asset(affine_model, key=affine_model_key, parent_asset=data_key)
    dataset.save()

else:
    affine_model = dataset.get_asset_data(affine_model_key)

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("******************************************")
print("Determine dimensionality for affine models")
t = time()
if redo_find_dim or not 'best_dim' in dataset.assets[affine_model_key]:
    best_dim = find_best_dimension(np.mat(data_train), labels_train)
    dataset.assets[affine_model_key]['best_dim'] = best_dim
    dataset.save()
else:
    best_dim = dataset.assets[affine_model_key]['best_dim']
print "best_dim: %i" % best_dim


dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
print("*******************")
print("Test classification")
t = time()
score = affine_model.score(np.mat(data_test), labels_test, dim=best_dim)
print "Accuricy: %.1f%%" % (100 * score)

dt = time() - t
print "Took %.3fs. (%.3fs)\n" % (time() - t0, dt)
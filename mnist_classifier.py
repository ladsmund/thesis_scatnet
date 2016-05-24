import argparse
import numpy as np
from utilities.dataset import Dataset
import scattconvnet
from classifiers.affine_model import AffineModel, reshape_to_1d, find_best_dimension

from time import time

parser = argparse.ArgumentParser()
parser.add_argument('mnist_path')
args = parser.parse_args()
mnist_path = args.mnist_path
dataset = Dataset(mnist_path)

redo_scatnet = False
redo_affine_model = False
redo_find_dim = False

labels_key = 'labels'
labels = dataset.get_asset_data(labels_key)

print("****************************")
print("Process ScatNet Coefficients")
data_key = 'scatnet_a06_s03_m02_coef_original'
if redo_scatnet or not data_key in dataset.assets:
    data = scattconvnet.process_data(dataset=dataset, input_asset_key='original', nangles=6, scale=3, max_depth=2)
    dataset.save()
else:
    data = dataset.get_asset_data(data_key)
data = reshape_to_1d(data)

print("*******************************")
print("Select test and train data sets")
data_train = data[:600:1]
data_test = data[600::1]
labels_train = labels[:600:1]
labels_test = labels[600::1]

print("************************************")
print("Generate affine model for all labels")
max_dim = 40
affine_model_key = 'affine_model' + data_key
if redo_affine_model or not affine_model_key in dataset.assets:
    affine_model = AffineModel(n_components=max_dim)
    affine_model.fit(data_train, labels_train)
    dataset.add_pickle_asset(affine_model, key=affine_model_key, parent_asset=data_key)
    dataset.save()

else:
    affine_model = dataset.get_asset_data(affine_model_key)

print("******************************************")
print("Determine dimensionality for affine models")
if redo_find_dim or not 'best_dim' in dataset.assets[affine_model_key]:
    best_dim = find_best_dimension(np.mat(data_train), labels_train)
    dataset.assets[affine_model_key]['best_dim'] = best_dim
    dataset.save()
else:
    best_dim = dataset.assets[affine_model_key]['best_dim']
print "best_dim: %i" % best_dim


print("*******************")
print("Test classification")
score = affine_model.score(np.mat(data_test), labels_test, dim=best_dim)
print "Accuricy: %.1f%%" % (100 * score)

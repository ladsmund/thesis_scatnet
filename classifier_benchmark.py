import argparse
import numpy as np
from utilities.dataset import Dataset
import scattconvnet
from classifiers.affine_model import AffineModel, reshape_to_1d, find_best_dimension
from classifiers.affine_model_old import AffineModelOld
from classifiers.affine_model_multi import AffineModelPar

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
data_train = np.mat(data[:train_test_split])
data_test = np.mat(data[train_test_split:])
labels_train = labels[:train_test_split]
labels_test = labels[train_test_split:]


print "Affine Model Old"
a = AffineModelOld()
t = time()
a.fit(data_train, labels_train)
t2 = time()
s1 = a.score(data_test, labels_test, dim=10)
dt2 = time() - t2
dt = time() - t
print "score: %.4f" % s1
print "Took %5.3fs %5.3fs\n" % (dt, dt2)

print "Affine Model New"
a = AffineModel()
t = time()
a.fit(data_train, labels_train)
t2 = time()
s2 = a.score(data_test, labels_test, dim=None)
dt2 = time() - t2
dt = t2 - t
print "score: %.4f" % s2
print "Took %5.3fs %5.3fs\n" % (dt, dt2)

print "Affine Model Par"
a = AffineModelPar()
t = time()
a.fit(data_train, labels_train)
t2 = time()
s3 = a.score(data_test, labels_test, dim=10)
dt2 = time() - t2
dt = t2 - t
print "score: %.4f" % s3
print "Took %5.3fs %5.3fs\n" % (dt, dt2)





exit()


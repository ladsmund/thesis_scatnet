
from scattconvnet.wavelet_transform import ScatNet
from classifiers.affine_model import AffineModel
from sklearn.datasets import fetch_lfw_people

lfw_people = fetch_lfw_people(min_faces_per_person=50)

scatnet = ScatNet(4,2,2)



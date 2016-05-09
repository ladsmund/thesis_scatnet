
import numpy as np
# import scipy
import scipy.signal
from time import time

import cv2
from scattconvnet.wavelet_transform import generate_kernels, wavelet_transform, scatt_coefficients

img = scipy.misc.ascent()

img = cv2.imread("./data/test_img.tiff")
img = img.astype('float64')
# img = img[::2, ::2]

n_angles = 4
J = 3

t = time()
kernel_layers = generate_kernels(4, J)
print "Generate: %.0f ms" % ((time() - t) * 1000)

t = time()
results = wavelet_transform(img, kernel_layers)
print "Scattering transform: %.0f ms" % ((time() - t) * 1000)


t = time()
scattering_coefficients = scatt_coefficients(results, J)
print "Scattering coefficients: %.0f ms" % ((time() - t) * 1000)

print scattering_coefficients.dtype

# cv2.imwrite("scat_coeff.tiff", scattering_coefficients)
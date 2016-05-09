import numpy as np
import cv2

from wavelet import morlet, gauss_kernel
from utilities.dataset.dataset import Dataset
from utilities.dataset.scat_response import ScatResponse
from multiprocessing import Pool

from time import time

DEFAULT_SCALE = 4
DEFAULT_MAX_DEPTH = 2
DEFAULT_NANGLES = 4

def conv(img, kernel):
    r = cv2.filter2D(img, -1, np.real(kernel))
    i = cv2.filter2D(img, -1, np.imag(kernel))
    return np.sqrt(r ** 2 + i ** 2)


class ScatNet:
    def __init__(self, nangles, scale, max_depth=DEFAULT_MAX_DEPTH):
        self.nangles = nangles
        self.scale = scale
        self.max_depth = max_depth
        self.kernel_layers = self.generate_kernels(nangles, scale)

    def transform(self, img):
        responses = self.wavelet_transform(img, self.kernel_layers, self.max_depth)
        return self.scatt_coefficients(responses, self.scale)

    @staticmethod
    def wavelet_transform(src, kernel_layers, max_depth=DEFAULT_MAX_DEPTH):
        responses = [src]

        if len(kernel_layers) == 0 or max_depth == 0:
            return responses

        while len(kernel_layers) > 0:
            kernels = kernel_layers[0]
            for kernel in kernels:
                response = conv(src, kernel)
                responses += ScatNet.wavelet_transform(response, kernel_layers[1:], max_depth - 1)
            kernel_layers = kernel_layers[:-1]

        return responses

    @staticmethod
    def scatt_coefficients(responses, J):
        blur_kernel = gauss_kernel(2 ** J)
        downsample_step = 2 ** (J - 1)

        def scat_coeff(r):
            sc = cv2.filter2D(r, -1, blur_kernel)
            return sc[::downsample_step, ::downsample_step]

        scattering_coefficients = [scat_coeff(r) for r in responses]
        return np.concatenate(scattering_coefficients)

    @staticmethod
    def generate_kernels(n_angles, J):
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        sigmas = 2 ** (np.arange(1, J) + 1)

        kernel_layers = list()

        for sigma in sigmas:
            kernels = list()
            for angle in angles:
                kernels.append(morlet(sigma, angle))
            kernel_layers.append(kernels)

        return kernel_layers


def init_worker_process(kernel_layers_arg, scale_arg):
    global scale
    scale = scale_arg
    global kernel_layers
    kernel_layers = kernel_layers_arg

def worker(img):
    global scale
    global kernel_layers
    responses = ScatNet.wavelet_transform(img, kernel_layers)
    coefficient = ScatNet.scatt_coefficients(responses, scale)
    return (responses, coefficient)

def process_data(dataset, scale=DEFAULT_SCALE, nangles=DEFAULT_NANGLES, max_depth=DEFAULT_MAX_DEPTH):
    # assert (isinstance(dataset, Dataset))
    train_data, train_labels = dataset.get_data()
    test_data, test_labels = dataset.get_data(test=True)

    nimages = len(train_data) + len(test_data)
    kernel_layers = ScatNet.generate_kernels(nangles, scale)

    pool = Pool(processes=4, initializer=init_worker_process, initargs=(kernel_layers, scale, ))

    t_total = time()

    train_responses, train_coefficients = zip(*pool.map(worker, train_data))
    test_responses, test_coefficients = zip(*pool.map(worker, test_data))
    print "Process %i images in %.0f ms" % (nimages, (time() - t_total) * 1000)

    # train_responses = []
    # train_coefficients = []
    # for img in train_data:
    #     responses = ScatNet.wavelet_transform(img, kernel_layers)
    #     train_responses.append(responses)
    #     train_coefficients.append(ScatNet.scatt_coefficients(responses, scale))
    #
    # print "%i train images in %.0f ms" % (len(train_data), (time() - t_total) * 1000)
    #
    # test_responses = []
    # test_coefficients = []
    # for img in test_data:
    #     responses = ScatNet.wavelet_transform(img, kernel_layers)
    #     test_responses.append(responses)
    #     test_coefficients.append(ScatNet.scatt_coefficients(responses, scale))
    # print "Process %i images in %.0f ms" % (nimages, (time() - t_total) * 1000)

    return ScatResponse(train_data=train_coefficients,
                        train_labels=train_labels,
                        train_response=train_responses,
                        test_data=test_coefficients,
                        test_labels=test_labels,
                        test_response=test_responses,
                        config={'nangles': nangles, 'max_depth': max_depth, 'scale': scale},
                        parent_dataset=dataset.parent_dataset + [dataset.__class__])


if __name__ == '__main__':
    import argparse
    from time import time
    import os

    t_total = time()

    parser = argparse.ArgumentParser("Scattering Wavelet Transformation")
    parser.add_argument('images', type=str, nargs='+')
    parser.add_argument('-J', '--scale', type=int, default=DEFAULT_SCALE)
    parser.add_argument('-a', '--nangles', type=int, default=DEFAULT_NANGLES)
    parser.add_argument('-m', '--maxdepth', type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    kernel_layers = ScatNet.generate_kernels(args.nangles, args.scale)

    for img_path in args.images:
        t = time()

        img_name = os.path.basename(img_path)
        feature_name = "%s" % (img_name.split('.')[0])

        img = cv2.imread(img_path)[:, :, 0]
        img = img.astype('float64')
        responses = ScatNet.wavelet_transform(img, kernel_layers)
        scattering_coefficients = ScatNet.scatt_coefficients(responses, args.scale)
        cv2.imwrite(os.path.join(args.output, img_name), scattering_coefficients)
        np.save(os.path.join(args.output, feature_name), scattering_coefficients)
        print "%s in %.0f ms" % (img_name, (time() - t) * 1000)

    print "Process %i images in %.0f ms" % (len(args.images), (time() - t_total) * 1000)

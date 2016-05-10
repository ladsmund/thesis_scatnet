import sys
from time import time

import numpy as np
import cv2
from multiprocessing import Pool

from wavelet import morlet, gauss_kernel
from utilities.dataset.dataset import Dataset, load_dataset, save_dataset
from utilities.dataset.scat_response import ScatResponse

DEFAULT_SCALE = 4
DEFAULT_MAX_DEPTH = 2
DEFAULT_NANGLES = 4

DEFAULT_N_PROCESS = 4


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


def process_data(dataset, scale=DEFAULT_SCALE, nangles=DEFAULT_NANGLES, max_depth=DEFAULT_MAX_DEPTH,
                 nprocess=DEFAULT_N_PROCESS):
    nimages = len(dataset.data)
    kernel_layers = ScatNet.generate_kernels(nangles, scale)

    pool = Pool(processes=nprocess, initializer=init_worker_process, initargs=(kernel_layers, scale,))

    t_total = time()

    responses, coefficients = zip(*pool.map(worker, dataset.data))
    print "Process %i images in %.0f ms" % (nimages, (time() - t_total) * 1000)

    return ScatResponse(data=coefficients,
                        responses=responses,
                        labels=dataset.labels,
                        mask_test=dataset.mask_test,
                        mask_train=dataset.mask_train,
                        config={'nangles': nangles, 'max_depth': max_depth, 'scale': scale},
                        parent_dataset=dataset.parent_dataset + [dataset.__class__])


if __name__ == '__main__':
    import argparse
    from time import time
    import os

    t_total = time()

    parser = argparse.ArgumentParser("Scattering Wavelet Transformation")
    parser.add_argument('inputs', type=str, nargs='+')
    parser.add_argument('-d', '--dataset_input', action='store_true', default=False)
    parser.add_argument('-J', '--scale', type=int, default=DEFAULT_SCALE)
    parser.add_argument('-a', '--nangles', type=int, default=DEFAULT_NANGLES)
    parser.add_argument('-m', '--maxdepth', type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument('-p', '--nprocesses', type=int, default=DEFAULT_N_PROCESS)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    print args

    if args.dataset_input:
        dataset = load_dataset(args.inputs[0])

        if not os.path.exists(os.path.dirname(args.output)):
            sys.stderr.write("No such directory: %s\n" % os.path.dirname(args.output))
            exit(1)

        res = process_data(dataset=dataset,
                           scale=args.scale,
                           nangles=args.nangles,
                           max_depth=args.maxdepth,
                           nprocess=args.nprocesses)

        save_dataset(res, args.output);
        pass


    else:
        kernel_layers = ScatNet.generate_kernels(args.nangles, args.scale)

        for img_path in args.inputs:
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

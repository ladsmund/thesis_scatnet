from time import time
import numpy as np
import cv2
from multiprocessing import Pool

from wavelet import morlet, gauss_kernel
from utilities.dataset import Dataset

DEFAULT_SCALE = 4
DEFAULT_MAX_DEPTH = 2
DEFAULT_NANGLES = 4
DEFAULT_N_PROCESS = None


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

        return np.array(responses)

    @staticmethod
    def scatt_coefficients(responses, J):
        blur_kernel = gauss_kernel(2 ** J)
        downsample_step = 2 ** (J - 1)

        blur = map(lambda i: conv(i, blur_kernel), responses)
        scatt_coeff = map(lambda i: i[::downsample_step, ::downsample_step], blur)

        return np.array(scatt_coeff)

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


def process_data(data, scale=DEFAULT_SCALE, nangles=DEFAULT_NANGLES, max_depth=DEFAULT_MAX_DEPTH,
                 nprocess=DEFAULT_N_PROCESS):
    nimages = data.shape[0]
    kernel_layers = ScatNet.generate_kernels(nangles, scale)

    pool = Pool(processes=nprocess, initializer=init_worker_process, initargs=(kernel_layers, scale,))

    t0 = time()

    scatt_res = pool.map(worker, data)
    responses, coefficients = zip(*scatt_res)
    responses = np.array(responses)
    coefficients = np.array(coefficients)

    print "type(responses): " + str(type(responses))
    t = (time() - t0) * 1000
    print "Process %i images in %.0f ms (%.0f ms / image)" % (nimages, t, t / nimages)

    return responses, coefficients


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Scattering Wavelet Transformation")
    parser.add_argument('inputs', type=str, nargs='+')
    parser.add_argument('-d', '--dataset_input', action='store_true', default=False)
    parser.add_argument('-k', '--asset_key', type=str)
    parser.add_argument('-J', '--scale', type=int, default=DEFAULT_SCALE)
    parser.add_argument('-a', '--nangles', type=int, default=DEFAULT_NANGLES)
    parser.add_argument('-m', '--maxdepth', type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument('-p', '--nprocesses', type=int, default=DEFAULT_N_PROCESS)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    dataset = Dataset(args.inputs[0])
    data = dataset.get_asset_data(args.asset_key)

    parameters = {"nangles": args.nangles,
                  "max_depth": args.maxdepth,
                  "scale": args.scale}

    responses, coefficients = process_data(data=data,
                                           scale=args.scale,
                                           nangles=args.nangles,
                                           max_depth=args.maxdepth,
                                           nprocess=args.nprocesses)

    generator_name = "scattnet"
    parameter_string = "a%02i_s%02i_m%02i" % (args.nangles, args.scale, args.maxdepth)
    responses_key = "%s_%s_resp" % (generator_name, parameter_string)
    coefficients_key = "%s_%s_coef" % (generator_name, parameter_string)

    dataset.add_asset(responses, responses_key, generator=generator_name, parameters=parameters,
                      parent_asset=args.asset_key)
    dataset.add_asset(coefficients, coefficients_key, generator=generator_name, parameters=parameters,
                      parent_asset=args.asset_key)

    dataset.save()

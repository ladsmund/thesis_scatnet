from time import time
import numpy as np
import cv2
from multiprocessing import Pool

import sys

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
    def __init__(self, nangles, scale, max_order=DEFAULT_MAX_DEPTH):
        self.nangles = nangles
        self.angles = np.linspace(0, np.pi, nangles, endpoint=False)
        self.scale = scale
        self.scales = 2 ** np.arange(1, scale + 1)
        self.max_order = max_order
        self.kernel_layers = self.generate_kernels(self.angles, self.scales)
        self.configs = self.scatter_config(range(self.nangles), range(self.scale), self.max_order)
        self.config_indices = dict(zip(self.configs, range(len(self.configs))))

    def transform(self, img):
        responses = self.wavelet_transform(img)
        return responses, self.scatt_coefficients(responses, self.scale)

    def wavelet_transform(self, src_org):
        res = np.zeros((len(self.configs),) + src_org.shape, dtype='float32')

        for c in self.configs:
            conv_config = c[:-1]
            index = self.config_indices[c]

            if len(conv_config) is 0:
                res[index, :, :] = src_org
                continue

            angle_indx, scale_indx = c[-1]
            kernel = self.kernel_layers[angle_indx][scale_indx]

            src_indx = self.config_indices[c[:-1]]
            src = res[src_indx]
            res[index, :, :] = conv(src, kernel)

        return res

    @staticmethod
    def scatt_coefficients(responses, J):
        blur_kernel = gauss_kernel(2 ** J)
        downsample_step = 2 ** (J - 1)

        blur = map(lambda i: conv(i, blur_kernel), responses)
        scatt_coeff = map(lambda i: i[::downsample_step, ::downsample_step], blur)

        return np.array(scatt_coeff)

    @staticmethod
    def generate_kernels(angles, scales):
        kernel_layers = list()
        for sigma in scales:
            kernels = list()
            for angle in angles:
                kernels.append(morlet(sigma, angle))
            kernel_layers.append(kernels)

        return kernel_layers

    @staticmethod
    def scatter_config(angles, scales, max_order, config=[(None,)]):
        if max_order <= 0:
            return config

        config_layer = []
        for si, s in enumerate(scales):
            config_new = []
            for a in angles:
                config_new += [c + ((s, a),) for c in config]
            config_layer += ScatNet.scatter_config(angles, scales[si + 1:], max_order - 1, config_new)

        return config + config_layer


def proc_instantiater(scatnet):
    global proc_scatnet
    proc_scatnet= scatnet

def proc_worker(img):
    global proc_scatnet
    return proc_scatnet.transform(img)


def process_data(data, scale=DEFAULT_SCALE, nangles=DEFAULT_NANGLES, max_depth=DEFAULT_MAX_DEPTH,
                 multi_process=True):
    nimages = data.shape[0]

    t0 = time()

    scatnet = ScatNet(nangles=nangles, scale=scale, max_order=max_depth)

    if multi_process:
        pool = Pool(initializer=proc_instantiater, initargs=(scatnet,))
        scatt_res = pool.map(proc_worker, data)
    else:
        scatt_res = map(scatnet.transform, data)

    responses, coefficients = zip(*scatt_res)
    responses = np.array(responses)
    coefficients = np.array(coefficients)

    t = (time() - t0) * 1000
    print "Processed %i images in %.0f ms (%.0f ms / image)" % (nimages, t, t / nimages)

    return responses, coefficients


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Scattering Wavelet Transformation")
    parser.add_argument('inputs', type=str, nargs='+')
    parser.add_argument('-i', '--image_input', action='store_true', default=False)
    parser.add_argument('-k', '--asset_key', type=str)
    parser.add_argument('-J', '--scale', type=int, default=DEFAULT_SCALE)
    parser.add_argument('-a', '--nangles', type=int, default=DEFAULT_NANGLES)
    parser.add_argument('-m', '--maxdepth', type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument('-p', '--multi_process', action='store_false', default=True)
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
                                           multi_process=args.multi_process)

    generator_name = "scattnet"
    parameter_string = "a%02i_s%02i_m%02i" % (args.nangles, args.scale, args.maxdepth)
    responses_key = "%s_%s_resp" % (generator_name, parameter_string)
    coefficients_key = "%s_%s_coef" % (generator_name, parameter_string)

    dataset.add_asset(responses, responses_key, generator=generator_name, parameters=parameters,
                      parent_asset=args.asset_key)
    dataset.add_asset(coefficients, coefficients_key, generator=generator_name, parameters=parameters,
                      parent_asset=args.asset_key)

    dataset.save()

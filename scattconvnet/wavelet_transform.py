from time import time, sleep
import numpy as np
import cv2
import multiprocessing
from multiprocessing import Pool

import sys
import os
import signal
import tempfile

from wavelet import morlet, gauss_kernel
from utilities.dataset import Dataset

DEFAULT_SCALE = 4
DEFAULT_MAX_DEPTH = 2
DEFAULT_NANGLES = 4
DEFAULT_N_PROCESS = None

DEFAULT_CALLBACK_INTERVAL = .5


def conv(img, kernel):
    if kernel.dtype == 'complex':
        r = cv2.filter2D(img, -1, np.real(kernel))
        i = cv2.filter2D(img, -1, np.imag(kernel))
        return r + i * 1j
    else:
        return cv2.filter2D(img, -1, kernel)


class ScatNet:
    def __init__(self, nangles, scale, max_order=DEFAULT_MAX_DEPTH):
        self.nangles = nangles
        self.angles = np.linspace(3 * np.pi / 2, np.pi / 2, nangles, endpoint=False)
        self.scale = scale
        self.scales = 2 ** np.arange(0, scale)
        self.max_order = max_order
        self.kernel_layers = self.generate_kernels(self.angles, self.scales)
        self.configs = self.scatter_config(range(self.nangles), range(self.scale), self.max_order)
        self.out_dim = len(self.configs)

        def get_key(c):
            l = len(c)
            if l == 0:
                return l
            else:
                a, s = zip(*c)
                return l, s, a

        self.configs.sort(key=get_key)

        self.config_indices = dict(zip(self.configs, range(self.out_dim)))
        self.blur_kernel = gauss_kernel(self.scales[-1])
        self.downsample_step = 2 ** (self.scale - 1)

    def transform(self, img):
        responses = self.wavelet_transform(img)
        return responses, self.scatt_coefficients(responses)

    def wavelet_transform(self, src_org):
        res = np.zeros((len(self.configs),) + src_org.shape, dtype='float64')

        for c in self.configs:
            index = self.config_indices[c]

            if len(c) is 0:
                res[index, :, :] = src_org
                continue

            scale_indx, angle_indx = c[-1]
            kernel = self.kernel_layers[angle_indx][scale_indx]
            src_indx = self.config_indices[c[:-1]]
            src = res[src_indx]

            res[index, :, :] = np.abs(conv(src, kernel))

        return res

    def scatt_coefficients(self, responses):
        blur = map(lambda i: conv(i, self.blur_kernel), responses)
        scatt_coeff = map(lambda i: i[::self.downsample_step, ::self.downsample_step], blur)
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
    def scatter_config(angles, scales, max_order, config=[()]):
        if max_order <= 0:
            return config

        config_layer = []
        for si, s in enumerate(scales):
            config_new = []
            for a in angles:
                config_new += [c + ((a, s),) for c in config]
            config_layer += ScatNet.scatter_config(angles, scales[si + 1:], max_order - 1, config_new)

        return config + config_layer


def proc_instantiater(scatnet, status_counter_arg, data_arg, responses_array, coefficients_array):
    global data
    data = data_arg
    global responses
    responses = responses_array
    global coefficients
    coefficients = coefficients_array
    global proc_scatnet
    proc_scatnet = scatnet
    global status_counter
    status_counter = status_counter_arg


def proc_worker(index):
    global proc_scatnet
    global status_counter
    global data
    global responses
    global coefficients
    response = proc_scatnet.wavelet_transform(data[index, :, :])
    coefficients[index, :, :, :] = proc_scatnet.scatt_coefficients(response)
    responses[index, :, :, :] = response
    with status_counter.get_lock():
        status_counter.value += 1


def print_status(proc_count, proc_size, start_time):
    if proc_count > 0:
        avg_speed = 1000 * (time() - start_time) / proc_count
    else:
        avg_speed = 0
    s = "%6i/%i - %5.1fms/img\n" % (proc_count, proc_size, avg_speed)
    sys.stdout.write(s)


def process_data(data, scale=DEFAULT_SCALE, nangles=DEFAULT_NANGLES, max_depth=DEFAULT_MAX_DEPTH,
                 multi_process=True, status_callback=print_status, callback_interval=DEFAULT_CALLBACK_INTERVAL,
                 responses_file=None, coefficients_file=None):
    nimages = data.shape[0]

    t0 = time()
    print "Instantiate scatnet"
    scatnet = ScatNet(nangles=nangles, scale=scale, max_order=max_depth)


    print "Instantiate output arrays"
    if responses_file is None:
        responses_file = tempfile.NamedTemporaryFile(mode='w+')
    elif isinstance(responses_file, str):
        responses_file = open(responses_file, 'w+')
    responses = np.memmap(responses_file, mode='w+',
                          shape=(data.shape[0], scatnet.out_dim, data.shape[1], data.shape[2]),
                          dtype='float64'
                          )
    if coefficients_file is None:
        coefficients_file = tempfile.NamedTemporaryFile(mode='w+')
    elif isinstance(coefficients_file, str):
        coefficients_file = open(coefficients_file, 'w+')
    coefficients = np.memmap(coefficients_file, mode='w+',
                             shape=(data.shape[0], scatnet.out_dim, data.shape[1] // scatnet.downsample_step,
                                    data.shape[2] // scatnet.downsample_step),
                             dtype='float64'
                             )

    if multi_process:
        print "Instantiate process pool"
        status_counter = multiprocessing.Value('i', 0)
        pool = Pool(initializer=proc_instantiater, initargs=(scatnet, status_counter, data, responses, coefficients))
        print "Start processing"
        defer = pool.map_async(proc_worker, range(nimages))

        while True:
            status = status_counter.value
            status_callback(status, nimages, t0)
            if status >= nimages:
                break
            sleep(callback_interval)
        scatt_res = defer.get()
        pool.close()
    else:
        status_callback(0, nimages, t0)
        scatt_res = map(scatnet.transform, data)
        status_callback(nimages, nimages, t0)

        responses, coefficients = zip(*scatt_res)
        responses = np.array(responses)
        coefficients = np.array(coefficients)

    t1 = time()
    responses.flush()
    coefficients.flush()
    print("Flushing took %.0f ms\n" % (1000 * (time() - t1)))

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
    parser.add_argument('-p', '--multi_process', action='store_true', default=False)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    dataset = Dataset(args.inputs[0])
    data = dataset.get_asset_data(args.asset_key)

    parameters = {"nangles": args.nangles,
                  "max_depth": args.maxdepth,
                  "scale": args.scale}

    generator_name = "scattnet"
    parameter_string = "a%02i_s%02i_m%02i" % (args.nangles, args.scale, args.maxdepth)
    responses_key = "%s_%s_%s_resp" % (generator_name, parameter_string, args.asset_key)
    coefficients_key = "%s_%s_%s_coef" % (generator_name, parameter_string, args.asset_key)

    signal.signal(signal.SIGINT, lambda signum, _: exit(signum))
    responses, coefficients = process_data(data=data,
                                           scale=args.scale,
                                           nangles=args.nangles,
                                           max_depth=args.maxdepth,
                                           multi_process=args.multi_process,
                                           responses_file=os.path.join(dataset.path, responses_key + ".data"),
                                           coefficients_file=os.path.join(dataset.path, coefficients_key + ".data"), )

    dataset.add_asset(responses, responses_key, generator=generator_name, parameters=parameters,
                      parent_asset=args.asset_key)
    dataset.add_asset(coefficients, coefficients_key, generator=generator_name, parameters=parameters,
                      parent_asset=args.asset_key)

    dataset.save()

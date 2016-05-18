from time import time, sleep
import numpy as np
import multiprocessing
from multiprocessing import Pool

import sys
import os
import signal
import tempfile

from scatnet import ScatNet, DEFAULT_SCALE, DEFAULT_NANGLES, DEFAULT_MAX_DEPTH
from utilities.dataset import Dataset

DEFAULT_CALLBACK_INTERVAL = .5


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
                          shape=(data.shape[0], scatnet.feature_dimension, data.shape[1], data.shape[2]),
                          dtype='float32'
                          )
    if coefficients_file is None:
        coefficients_file = tempfile.NamedTemporaryFile(mode='w+')
    elif isinstance(coefficients_file, str):
        coefficients_file = open(coefficients_file, 'w+')
    coefficients = np.memmap(coefficients_file, mode='w+',
                             shape=(data.shape[0], scatnet.feature_dimension, data.shape[1] // scatnet.downsample_step,
                                    data.shape[2] // scatnet.downsample_step),
                             dtype='float32'
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

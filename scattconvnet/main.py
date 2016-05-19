from time import time
import itertools

import sys
import os
import signal

from multi_process import process
from scatnet import ScatNet, DEFAULT_SCALE, DEFAULT_NANGLES, DEFAULT_MAX_DEPTH
from utilities.dataset import Dataset, parse_asset_path

DEFAULT_CALLBACK_INTERVAL = .5


def print_status(proc_count, proc_size, start_time):
    if proc_count > 0:
        avg_speed = 1000 * (time() - start_time) / proc_count
    else:
        avg_speed = 0
    s = "%6i/%i - %5.1fms/img\n" % (proc_count, proc_size, avg_speed)
    sys.stdout.write(s)


def process_data(dataset,
                 asset_key,
                 scale=DEFAULT_SCALE,
                 nangles=DEFAULT_NANGLES,
                 max_depth=DEFAULT_MAX_DEPTH,
                 multi_process=True,
                 status_callback=print_status,
                 callback_interval=DEFAULT_CALLBACK_INTERVAL):
    print "Instantiate scatnet"
    t0 = time()
    scatnet = ScatNet(nangles=nangles, scale=scale, max_order=max_depth)

    input = dataset.get_asset_data(asset_key)
    nimages = input.shape[0]
    input_shape = input.shape[1:]

    #######################################################
    print "Instantiate output arrays"
    #######################################################
    if scatnet.return_response:
        key = "scatnet_%s_%s_resp" % (scatnet.get_config_string(), asset_key)
        shape = scatnet.response_shape(input_shape, nimages)
        responses = dataset.new_asset(key=key, dtype=scatnet.response_dtype, shape=shape, generator='scatnet',
                                      parent_asset=asset_key)
    else:
        responses = None
    key = "scatnet_%s_%s_coef" % (scatnet.get_config_string(), asset_key)
    shape = scatnet.coefficient_shape(input_shape, nimages)
    coefficients = dataset.new_asset(key=key, dtype=scatnet.coefficient_dtype, shape=shape, generator='scatnet',
                                     parent_asset=asset_key)

    #######################################################
    print "Start processing"
    #######################################################
    if multi_process:
        iterator = process(scatnet, input)
    else:
        iterator = enumerate(itertools.imap(scatnet.transform, input))

    callback_time = time()
    for i, res in iterator:
        if responses is not None:
            coefficients[i] = res[0]
            responses[i] = res[1]
        else:
            coefficients[i] = res
        if time() - callback_time > callback_interval:
            callback_time = time()
            status_callback(i, nimages, t0)

    t1 = time()
    if responses:
        responses.flush()
    coefficients.flush()
    print("Flushing took %.0f ms" % (1000 * (time() - t1)))

    t = (time() - t0) * 1000
    print "Processed %i images in %.0f ms (%.0f ms / image)" % (nimages, t, t / nimages)

    return coefficients, responses


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Scattering Wavelet Transformation")
    parser.add_argument('inputs', type=str, nargs='+')
    parser.add_argument('-i', '--image_input', action='store_true', default=False)
    parser.add_argument('-k', '--asset_key', type=str, default=None)
    parser.add_argument('-J', '--scale', type=int, default=DEFAULT_SCALE)
    parser.add_argument('-a', '--nangles', type=int, default=DEFAULT_NANGLES)
    parser.add_argument('-m', '--maxdepth', type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument('-p', '--multi_process', action='store_true', default=False)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    if os.path.isfile(args.inputs[0]):
        asset_key, dataset = parse_asset_path(args.inputs[0])
    else:
        if args.asset_key is None:
            sys.stderr.write('asset key need to be given when nok specified in the path.\n')
            sys.stderr.flush()
            exit(1)
        asset_key = args.asset_key
        dataset = Dataset(args.inputs[0])

    print "asset_key: %s" % asset_key

    signal.signal(signal.SIGINT, lambda signum, _: exit(signum))
    process_data(dataset=dataset,
                 asset_key=asset_key,
                 scale=args.scale,
                 nangles=args.nangles,
                 max_depth=args.maxdepth,
                 multi_process=args.multi_process, )

    dataset.save()

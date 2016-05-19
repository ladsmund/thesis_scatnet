from time import time
import itertools

import sys

from multi_process import MultiScatNet
from scatnet import ScatNet, DEFAULT_SCALE, DEFAULT_NANGLES, DEFAULT_MAX_DEPTH
from utilities.dataset import Dataset, parse_asset_path

DEFAULT_CALLBACK_INTERVAL = .5


def process_data(input = None,
                 dataset = None,
                 input_asset_key = None,
                 scale=DEFAULT_SCALE,
                 nangles=DEFAULT_NANGLES,
                 max_depth=DEFAULT_MAX_DEPTH,
                 multi_process=True,
                 callback_interval=DEFAULT_CALLBACK_INTERVAL):
    print "Instantiate scatnet"
    t0 = time()

    if multi_process:
        scatnet = MultiScatNet(nangles=nangles, scale=scale, max_order=max_depth)
    else:
        scatnet = ScatNet(nangles=nangles, scale=scale, max_order=max_depth)

    input = dataset.get_asset_data(input_asset_key)
    nimages = input.shape[0]
    input_shape = input.shape[1:]

    #######################################################
    # Instantiate output arrays
    #######################################################
    key = "%s_%s" % (scatnet.get_key(), input_asset_key)
    shape = scatnet.get_shape(input_shape, nimages)
    output = dataset.new_asset(key=key, dtype=scatnet.coefficient_dtype, shape=shape, generator='scatnet',
                                     parent_asset=input_asset_key)

    #######################################################
    # Start processing
    #######################################################
    global last_write_time
    last_write_time = time()
    def print_status(proc_count, proc_size):
        global last_write_time
        if (time() - last_write_time) < callback_interval:
            return
        last_write_time = time()

        if proc_count > 0:
            avg_speed = 1000 * (time() - t0) / proc_count
        else:
            avg_speed = 0
        s = "%6i/%i - %5.1fms/img\n" % (proc_count, proc_size, avg_speed)
        sys.stdout.write(s)

    scatnet.process_images(input, output, status_callback=print_status)


    t1 = time()
    output.flush()
    print("Flushing took %.0f ms" % (1000 * (time() - t1)))

    t = (time() - t0) * 1000
    print "Processed %i images in %.0f ms (%.0f ms / image)" % (nimages, t, t / nimages)

    return output

import argparse
import os
import sys
import signal

from scattconvnet import process_data
from scattconvnet.utilities.dataset import parse_asset_path, Dataset
from scattconvnet.scatnet import DEFAULT_SCALE, DEFAULT_NANGLES, DEFAULT_MAX_DEPTH

if __name__ == '__main__':

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
                 input_asset_key=asset_key,
                 scale=args.scale,
                 nangles=args.nangles,
                 max_depth=args.maxdepth,
                 multi_process=args.multi_process, )

    dataset.save()

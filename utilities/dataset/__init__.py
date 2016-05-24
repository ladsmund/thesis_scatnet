import os
import sys

from dataset import Dataset

Dataset = Dataset


def asset_key_from_path(path):
    return os.path.basename(path).split('.')[0]


def parse_asset_path(path):
    dir = os.path.dirname(path)
    asset_key = asset_key_from_path(path)
    dataset = Dataset(dir)
    return asset_key, dataset


def parse_arg_path(args):
    if os.path.isfile(args.inputs[0]):
        asset_key, dataset = parse_asset_path(args.inputs[0])
    else:
        if args.asset_key is None:
            sys.stderr.write('asset key need to be given when nok specified in the path.\n')
            sys.stderr.flush()
            exit(1)
        asset_key = args.asset_key
        dataset = Dataset(args.inputs[0])
    return dataset, asset_key

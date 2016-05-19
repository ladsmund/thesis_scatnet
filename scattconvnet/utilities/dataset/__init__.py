import os

from dataset import Dataset

Dataset = Dataset

def parse_asset_path(path):
    dir = os.path.dirname(path)
    asset_key = os.path.basename(path).split('.')[0]
    dataset = Dataset(dir)
    return asset_key, dataset




import sklearn.datasets as datasets
from dataset import Dataset

def fetch_mldata(dist_path, data_name):
    skit_dataset = datasets.fetch_mldata(data_name)

    d = Dataset(dist_path)
    d.add_asset(skit_dataset.data, key='original', generator='source')
    d.add_asset(skit_dataset.target, key='labels', generator='source')
    d.save()
    return d

def fetch_mnist(dist_path):
    d = fetch_mldata(dist_path, 'MNIST original')
    d.assets['original']['shape'] = [70000,28,28]
    d.save()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Dataset fetcher")
    parser.add_argument('path', type=str)
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    path = args.path
    name = args.name

    name_map =dict()
    name_map['mnist'] = fetch_mnist

    if name in name_map:
        name_map[name](path)
    else:
        fetch_mldata(path, name)

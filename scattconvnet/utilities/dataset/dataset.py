import numpy as np
import json
import os
import shutil


def numpy_memmap(path, shape, dtype):
    return np.memmap(path, mode='r', shape=shape, dtype=dtype)


class Dataset:
    ASSET_EXT = "data"
    META_FILE_EXT = "json"

    def __init__(self, path, name=None, meta_file_path=None):
        self.path = path
        self.assets = dict()
        self.name = name
        if name is None:
            self.name = os.path.basename(path).split('.')[0]
        self._meta_file_path = meta_file_path
        if meta_file_path is None:
            self._meta_file_path = os.path.join(self.path, "%s.%s" % (self.name, self.META_FILE_EXT))

        if os.path.exists(self._meta_file_path):
            self.load()
        elif not os.path.exists(self.path):
            os.mkdir(self.path)

    def save(self, path=None):
        if path is None:
            path = self._meta_file_path
        j_string = json.dumps(self.dump(), indent=2, separators=(',', ': '), sort_keys=True)
        open(path, 'w').write(j_string)

    def load(self, path=None):
        if path is None:
            path = self._meta_file_path
        for k, v in json.load(open(path, 'r')).items():
            setattr(self, k, v)

    def dump(self):
        d = self.__dict__.copy()
        for k in d.keys():
            if k[0] == '_':
                del d[k]
        return d

    def get_asset_data(self, key, array_init_fun=numpy_memmap):
        if key not in self.assets:
            raise KeyError("No asset with key: %s" % str(key))
        path = self.asset_file_path(key)
        shape = tuple(self.assets[key]['shape'])
        dtype = self.assets[key]['dtype']
        return array_init_fun(path, shape, dtype)

    def new_asset(self, key, shape, dtype, generator, parameters=None, version=None, parent_asset=None):
        asset = dict()
        asset['generator'] = generator
        asset['version'] = version
        asset['parameters'] = parameters
        asset['parent_asset'] = parent_asset
        asset['shape'] = list(shape)
        asset['dtype'] = str(dtype)
        self.assets[key] = asset

        filename = self.asset_file_path(key)
        return np.memmap(filename, mode='w+', shape=shape, dtype=dtype)

    def add_asset(self, array, key, generator, version=None, parameters=None, parent_asset=None):
        # Write array to data file
        filename = os.path.join(self.path, key + self.ASSET_EXT)

        if isinstance(array, np.memmap) and array.filename == filename:
            # print "The memory map file was already written to the asset file"
            array.flush()
        else:
            data_file = open(filename, 'wb')
            data_file.write(array.tobytes())
            data_file.close()

        asset = dict()
        asset['generator'] = generator
        asset['version'] = version
        asset['parameters'] = parameters
        asset['parent_asset'] = parent_asset
        asset['shape'] = list(array.shape)
        asset['dtype'] = str(array.dtype)
        self.assets[key] = asset

    def asset_file_path(self, key):
        file_name = key + "." + self.ASSET_EXT
        return os.path.join(self.path, file_name)

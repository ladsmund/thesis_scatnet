import os

DATA_FOLDER_NAME = "data"
module_dir = os.path.dirname(__file__)
_data_folder = os.path.join(module_dir, DATA_FOLDER_NAME)
if not os.path.exists(_data_folder):
    print "mcl_util.dataset: Generate data folder"
    os.makedirs(_data_folder)


def get_path():
    return _data_folder

from dataset import Dataset

class ScatResponse(Dataset):

    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config')
        self.train_response = kwargs.pop('train_response')
        self.test_response = kwargs.pop('test_response')
        Dataset.__init__(self, *args, **kwargs)


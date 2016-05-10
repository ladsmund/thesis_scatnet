
from dataset import Dataset

class ScatResponse(Dataset):

    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config')
        self.responses = kwargs.pop('responses')
        Dataset.__init__(self, *args, **kwargs)


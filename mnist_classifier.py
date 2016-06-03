from classifiers.affine_model import AffineModel
from normalizer import Normalizer


class classifier(AffineModel):
    def __init__(self, **kwargs):
        self.normalizer = None
        AffineModel.__init__(self, **kwargs)

    def fit(self, data, *args, **kwargs):
        self.normalizer = Normalizer(data)
        return AffineModel.fit(self, data, *args, **kwargs)

    def score(self, data, *args, **kwargs):
        d = self.normalizer.normalize(data)
        return AffineModel.score(self, d, *args, **kwargs)

import numpy as np
import multiprocessing
import affine_model


class AffineModelPar(affine_model.AffineModel):
    def __init__(self, *argv, **kwargs):
        affine_model.AffineModel.__init__(self, *argv, **kwargs)

    def classify(self, data, dim=None):
        pool = multiprocessing.Pool()
        distances = np.zeros(shape=(data.shape[0], len(self.model_keys)), dtype='float32')

        assync_results = []
        for i, model_key in enumerate(self.model_keys):
            model_path = self.get_model_path(model_key)
            apply_res = pool.apply_async(affine_model.distance, (model_path, data), {'dim': dim})
            assync_results.append((i, apply_res))
        pool.close()

        for i, r in assync_results:
            distances[:, i] = r.get()
        return np.argmin(distances, axis=1)

import numpy as np
import cv2
from wavelet import morlet, gauss_kernel

DEFAULT_SCALE = 4
DEFAULT_MAX_DEPTH = 2
DEFAULT_NANGLES = 4


def conv(img, kernel):
    if kernel.dtype == 'complex':
        r = cv2.filter2D(img, -1, np.real(kernel))
        i = cv2.filter2D(img, -1, np.imag(kernel))
        return r + i * 1j
    else:
        return cv2.filter2D(img, -1, kernel)


def generate_kernels(angles, scales):
    kernel_layers = list()
    for sigma in scales:
        kernels = list()
        for angle in angles:
            kernels.append(morlet(sigma, angle))
        kernel_layers.append(kernels)
    return kernel_layers


def get_scatter_config(angles, scales, max_order):
    def scatter_config(angles, scales, max_order, config):
        if max_order <= 0:
            return config
        config_layer = []
        for si, s in enumerate(scales):
            config_new = []
            for a in angles:
                config_new += [c + ((a, s),) for c in config]
            config_layer += scatter_config(angles, scales[si + 1:], max_order - 1, config_new)
        return config + config_layer

    def get_sort_key(c):
        l = len(c)
        if l == 0:
            return l
        else:
            a, s = zip(*c)
            return l, s, a

    configs = scatter_config(angles, scales, max_order, [()])
    configs.sort(key=get_sort_key)
    config_indices = dict(zip(configs, range(len(configs))))
    return configs, config_indices


class ScatNet:
    def __init__(self, nangles, scale, max_order=DEFAULT_MAX_DEPTH):
        self.nangles = nangles
        self.angles = np.linspace(3 * np.pi / 2, np.pi / 2, nangles, endpoint=False)
        self.scale = scale
        self.scales = 2 ** np.arange(0, scale)
        self.max_order = max_order
        self.kernel_layers = generate_kernels(self.angles, self.scales)

        self.configs, self.config_indices = get_scatter_config(range(self.nangles), range(self.scale), self.max_order)
        self.feature_dimension = len(self.configs)

        self.blur_kernel = gauss_kernel(self.scales[-1])
        self.downsample_step = 2 ** (self.scale - 1)

    def transform(self, img):
        responses = self.wavelet_transform(img)
        return responses, self.scatt_coefficients(responses)

    def wavelet_transform(self, src_org):
        res = np.zeros((len(self.configs),) + src_org.shape, dtype='float32')
        for c in self.configs:
            index = self.config_indices[c]

            # The first output response is the input image
            if len(c) is 0:
                res[index, :, :] = src_org
                continue

            scale_indx, angle_indx = c[-1]

            # Select source data
            src_indx = self.config_indices[c[:-1]]
            src = res[src_indx]

            # Select filter kernel
            kernel = self.kernel_layers[angle_indx][scale_indx]

            res[index] = np.abs(conv(src, kernel))

        return res

    def scatt_coefficients(self, responses):
        blur = map(lambda i: conv(i, self.blur_kernel), responses)
        scatt_coeff = map(lambda i: i[::self.downsample_step, ::self.downsample_step], blur)
        return np.array(scatt_coeff)

    def get_coefficient_shape(self, input_shape):
        downsample_shape = (map(lambda x: x//self.downsample_step, input_shape))
        return self.get_response_shape(downsample_shape)

    def get_response_shape(self, input_shape):
        return (self.feature_dimension,) + tuple(input_shape)


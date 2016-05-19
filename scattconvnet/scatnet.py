import numpy as np
import cv2
from wavelet import morlet, gauss_kernel
from time import time
import itertools

DEFAULT_SCALE = 4
DEFAULT_MAX_DEPTH = 2
DEFAULT_NANGLES = 4
DEFAULT_DTYPE = 'float32'


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
    response_dtype = 'float32'
    coefficient_dtype = 'float32'

    def __init__(self,
                 nangles,
                 scale,
                 max_order=DEFAULT_MAX_DEPTH,
                 return_response=False,
                 dtype=DEFAULT_DTYPE):

        self.nangles = nangles
        self.angles = np.linspace(3 * np.pi / 2, np.pi / 2, nangles, endpoint=False)
        self.scale = scale
        self.scales = 2 ** np.arange(0, scale)
        self.max_order = max_order
        self.dtype = dtype
        self.return_response = return_response

        self.kernel_layers = generate_kernels(self.angles, self.scales)
        self.configs, self.config_indices = get_scatter_config(range(self.nangles), range(self.scale), self.max_order)
        self.feature_dimension = len(self.configs)

        self.blur_kernel = gauss_kernel(self.scales[-1])
        self.downsample_step = 2 ** (self.scale - 1)

    def get_shape(self, input_shape, number=None):
        if not self.return_response:
            input_shape = map(lambda x: x // self.downsample_step, input_shape)
        if number is None:
            return (self.feature_dimension,) + tuple(input_shape)
        else:
            return (number, self.feature_dimension,) + tuple(input_shape)

    def get_key(self):
        str = "scatnet_a%02i_s%02i_m%02i" % (self.nangles, self.scale, self.max_order)
        if self.return_response:
            return str + "_resp"
        else:
            return str + "_coef"

    def get_dtype(self):
        return self.dtype

    def process(self, src_org):
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

        if self.return_response:
            return res
        else:
            return self.coefficients(res)

    def coefficients(self, responses):
        blur = map(lambda i: conv(i, self.blur_kernel), responses)
        scatt_coeff = map(lambda i: i[::self.downsample_step, ::self.downsample_step], blur)
        return np.array(scatt_coeff)

    def _process_iterator(self, images):
        return enumerate(itertools.imap(self.process, images))

    def process_images(self, images, output=None, status_callback=None):
        nimages = images.shape[0]

        if output is None:
            shape = (nimages,) + self.get_shape(images.shape[1:])
            output = np.zeros(shape=shape, dtype=self.get_dtype())

        progress = 0
        for i, res in self._process_iterator(images):
            output[i] = res
            progress += 1
            if status_callback:
                status_callback(progress, nimages)

        return output

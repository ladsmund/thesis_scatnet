import numpy as np
import os
import matplotlib.pyplot as plt


def write_to_file(scatnet, dist_path):
    filter_set = scatnet.kernel_layers

    for l, layer in enumerate(filter_set):
        for f, filter in enumerate(layer):
            # filter = np.real(filter)
            shape_str = "x".join(map(str, filter.shape))
            dtype = str(filter.dtype)
            filename = "filter_l%i_f%i_%s_%s" % (l, f, shape_str, dtype)
            filter_path = os.path.join(dist_path, filename)

            file = open(filter_path, 'wb')
            file.write(filter.tostring())
            file.close()


def show(scatnet):
    w_max, h_max = 0, 0
    for l in scatnet.kernel_layers:
        for k in l:
            w, h = k.shape
            w_max = max(w, w_max)
            h_max = max(h, h_max)

    filters = None
    for l in scatnet.kernel_layers:
        layer = None
        for k in l:
            w, h = k.shape
            pad_w = tuple(map(int, map(lambda f: f((w_max - w) / 2.), [np.floor, np.ceil])))
            pad_h = tuple(map(int, map(lambda f: f((h_max - h) / 2.), [np.floor, np.ceil])))
            k = np.pad(k, (pad_w, pad_h), 'constant')

            if layer is None:
                layer = k
            else:
                layer = np.concatenate([layer, k], 1)
        if filters is None:
            filters = layer
        else:
            filters = np.concatenate([filters, layer], 0)

    plt.figure()
    plt.imshow(np.real(filters), interpolation='none')
    plt.grid(True)
    plt.figure()
    plt.imshow(np.imag(filters), interpolation='none')
    plt.grid(True)
    plt.figure()
    plt.imshow(scatnet.blur_kernel, interpolation='none')
    plt.grid(True)
    plt.show()

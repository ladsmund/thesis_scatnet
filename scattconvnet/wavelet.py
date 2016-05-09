import numpy as np

DEFAULT_SIZE = 4

radius = lambda x, y: np.sqrt(x ** 2 + y ** 2)
gauss = lambda r, s: np.exp(-r ** 2 / (2. + s ** 2))
project = lambda x, y, a: np.cos(a) * x + np.sin(a) * y
wave1d = lambda r, s: np.exp(2j * np.pi * r / s)
wave2d = lambda x, y, s, a: wave1d(project(x, y, a), s)

def mesh(size):
    n = size // 2
    return np.mgrid[-n:n, -n:n]


def gauss_kernel(sigma, size=None):
    if size is None:
        size = int(DEFAULT_SIZE*sigma)
    xs, ys = mesh(size)
    kernel = gauss(radius(xs, ys), sigma)
    return kernel / kernel.sum()

def morlet(sigma, angle, size=None):
    if size is None:
        size = int(DEFAULT_SIZE*sigma)

    xs, ys = mesh(size)
    carrier = wave2d(xs, ys, sigma, angle)
    envelope = gauss(radius(xs, ys), sigma)

    # Adjust to the zero mean wavelet constrain
    # wavelet = np.multiply(carrier, envelope)
    wavelet = carrier * envelope
    sum0 = np.sum(wavelet)
    sum_envelope = np.sum(envelope)
    c0 = sum0 / sum_envelope
    wavelet = np.multiply(carrier - c0, envelope)

    # Adjust such that the squared norm equals 1
    square_norm = np.sum(np.power(np.abs(wavelet), 2))
    c1 = 1. / np.sqrt(square_norm)
    wavelet = c1 * wavelet

    return wavelet


if __name__ == '__main__':
    pass

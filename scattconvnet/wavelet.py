import numpy as np

DEFAULT_SIZE = 8
DEFAULT_XI = 1  # Scale frequency

radius = lambda x, y: np.sqrt(x ** 2 + y ** 2)
gauss = lambda r, s: np.exp(-r ** 2 / (2. * s ** 2))
project = lambda x, y, a: np.cos(a) * x + np.sin(a) * y
wave1d = lambda r, s, xi=np.pi: np.exp(-1j * xi * r / s)
wave2d = lambda x, y, s, a, xi=np.pi: wave1d(project(x, y, a), s, xi)


def mesh(size):
    n = size // 2
    return np.mgrid[-n:n, -n:n]


def gauss_kernel(sigma, size=None):
    if size is None:
        size = int(DEFAULT_SIZE * sigma)
    xs, ys = mesh(size)
    kernel = gauss(radius(xs, ys), sigma)
    return kernel / np.sum(kernel)


def morlet(sigma, angle, size=None, angle_freq=DEFAULT_XI):
    if size is None:
        size = int(DEFAULT_SIZE * sigma)

    xs, ys = mesh(size)
    carrier = wave2d(xs, ys, sigma, angle, angle_freq)
    envelope = gauss(radius(xs, ys), sigma)

    # Normalize envolope such that it sums to 1.
    envelope /= np.sum(envelope)

    # Adjust to the zero mean wavelet constrain
    wavelet = carrier * envelope
    sum0 = np.sum(wavelet)
    sum_envelope = np.sum(envelope)
    beta = sum0 / sum_envelope
    wavelet = np.multiply(carrier - beta, envelope)

    # Adjust such that the squared norm equals 1
    # square_norm = np.sum(np.power(np.abs(wavelet), 2))
    # alpha = 1. / np.sqrt(square_norm)
    # wavelet = alpha * wavelet

    return wavelet


if __name__ == '__main__':
    pass

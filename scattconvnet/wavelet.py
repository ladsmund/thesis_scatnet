import numpy as np


def morlet(size, frequency, samplerate):
    index = np.mat(np.arange(0, size) - size / 2)
    freq = frequency / float(samplerate)
    envelope = np.exp(-np.power(index, 2.) * freq ** 2 / 2)
    carrier = np.exp(1j * index * freq * 2 * np.pi)

    # Adjust to the zero mean wavelet constrain
    wavelet0 = np.multiply(carrier, envelope)
    sum0 = np.sum(wavelet0)
    sum_envelope = np.sum(envelope)
    c0 = sum0 / sum_envelope
    wavelet1 = np.multiply(carrier - c0, envelope)

    # Adjust such that the squared norm equals 1
    square_norm = np.sum(np.power(np.abs(wavelet1), 2))
    c1 = 1. / np.sqrt(square_norm)
    wavelet2 = c1 * wavelet1

    return wavelet2

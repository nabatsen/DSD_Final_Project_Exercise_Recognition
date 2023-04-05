import numpy as np
from scipy.stats import skew, kurtosis


def spectral_energy_of_energy_vectors(X):
    res = np.zeros((X.shape[0], 6))
    for i in range(0, X.shape[1], 3):
        energy_vector = np.square(X[:, :, i]) + np.square(X[:, :, i + 1]) + np.square(X[:, :, i + 2])
        mean_energy = np.mean(np.square(np.abs(np.fft.fft(energy_vector, axis=1))), axis=1)
        res[:, int(round(i / 3))] = mean_energy
    return res


def spectral_energy_of_energy_vectors_rep(rep):
    res = np.zeros(int(rep.shape[0] / 3))
    for i in range(0, rep.shape[0], 3):
        energy_vector = np.square(rep[i, :]) + np.square(rep[i + 1, :]) + np.square(rep[i + 2, :])
        mean_energy = np.mean(np.square(np.abs(np.fft.fft(energy_vector))))
        res[(round(i / 3))] = mean_energy
    return res


def extract_features(X):
    mean_matrix = np.mean(X, axis=2)
    variance = np.var(X, axis=2)
    std_dev_matrix = np.std(X, axis=2)
    max_matrix = np.max(X, axis=2)
    min_matrix = np.min(X, axis=2)
    skewness_matix = skew(X, axis=2)
    kurtosis_matrix = kurtosis(X, axis=2)
    mean_spectral_energy = np.mean(np.abs(np.fft.fft(X, axis=2)), axis=2)
    spectral_energy_of_energy_vectors_matrix = spectral_energy_of_energy_vectors(X)
    return np.concatenate(
        (mean_matrix, variance, std_dev_matrix, max_matrix, min_matrix, skewness_matix, kurtosis_matrix,
         mean_spectral_energy, spectral_energy_of_energy_vectors_matrix), axis=1)


def extract_features_for_single_reading(reading):
    res = np.zeros((0))
    mean_matrix = np.mean(reading, axis=1)
    res = np.append(res, mean_matrix)
    variance = np.var(reading, axis=1)
    res = np.append(res, variance)
    std_dev_matrix = np.std(reading, axis=1)
    res = np.append(res, std_dev_matrix)
    max_matrix = np.max(reading, axis=1)
    res = np.append(res, max_matrix)
    min_matrix = np.min(reading, axis=1)
    res = np.append(res, min_matrix)
    skewness_matix = skew(reading, axis=1)
    res = np.append(res, skewness_matix)
    kurtosis_matrix = kurtosis(reading, axis=1)
    res = np.append(res, kurtosis_matrix)
    mean_spectral_energy = np.mean(np.abs(np.fft.fft(reading, axis=1)), axis=1)
    res = np.append(res, mean_spectral_energy)
    spectral_energy_of_energy_vectors_matrix = spectral_energy_of_energy_vectors_rep(reading)
    res = np.append(res, spectral_energy_of_energy_vectors_matrix)
    return res

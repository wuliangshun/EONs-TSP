# config.py
import numpy as np

n_channels = 7#6,7
F = 50e9  # Hz
gamma = 1.32e-3
alpha = 0.22 / 4.343e3  # dB/km to 1/m
beta2 = -21.7e-27
L = 80e3
Ns = 5
h = 6.63e-34
nu = 193.55e12
n_sp = 1.58

np.random.seed(0)
powers_dBm = np.random.uniform(-2, 2, n_channels)
powers_W = 10 ** ((powers_dBm - 30) / 10)
bandwidths = np.full(n_channels, 40e9)
psd = powers_W / bandwidths

# utils.py
import numpy as np
from gn_model import nsr_single
from config import n_channels, psd, bandwidths

def build_U_matrix():
    U = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                nsr_ij = nsr_single(psd[i], psd[j], bandwidths[i], bandwidths[j], abs(i - j))
                nsr_ji = nsr_single(psd[j], psd[i], bandwidths[j], bandwidths[i], abs(i - j))
                U[i, j] = max(nsr_ij, nsr_ji)
    return U

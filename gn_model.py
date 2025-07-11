# gn_model.py
import numpy as np
from config import F, gamma, alpha, beta2, L, Ns, h, nu, n_sp

def nsr_single(Gi, Gj, Di, Dj, dist_ij):
    xi = (np.pi**2 * Dj**2 * beta2) / alpha
    cross_term = Gj**2 * np.log((dist_ij * F + Dj / 2) / (dist_ij * F - Dj / 2))
    self_term = Gi**2 * np.arcsinh(np.abs(xi))
    G_nli = (3 * gamma**2 * Gi) / (2 * np.pi * alpha * np.abs(beta2)) * (self_term + cross_term)
    G_ase = (np.exp(alpha * L) - 1) * h * nu * n_sp * Ns
    snr = Gi / (G_ase + Ns * G_nli)
    return 1 / snr  # NSR

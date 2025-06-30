import numpy as np
from scipy.signal import correlation_lags, correlate
import time

def estimate_tdoa_cc(sig1: np.ndarray, sig2: np.ndarray, fs: float) -> tuple[float, float]:
    """
    Estima el TDOA usando correlación cruzada clásica.
    Retorna: (tdoa en segundos, tiempo de cómputo en segundos)
    """
    start_time = time.perf_counter()
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()
    if len(sig1) == 0 or len(sig2) == 0:
        return np.nan, time.perf_counter() - start_time
    cc = correlate(sig1, sig2, mode='full')
    lags = correlation_lags(len(sig1), len(sig2), mode='full')
    tdoa_index = np.argmax(cc)
    tdoa = lags[tdoa_index] / fs
    return tdoa, time.perf_counter() - start_time

def estimate_tdoa_gcc(sig1: np.ndarray, sig2: np.ndarray, fs: float, method: str = "phat") -> tuple[float, float]:
    """
    Estima el TDOA usando GCC (Generalized Cross-Correlation).
    method: 'phat', 'roth', 'scot', etc.
    Retorna: (tdoa en segundos, tiempo de cómputo en segundos)
    """
    start_time = time.perf_counter()
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()
    n = len(sig1) + len(sig2) - 1
    SIG1 = np.fft.fft(sig1, n=n)
    SIG2 = np.fft.fft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    if method == "phat":
        denom = np.abs(R)
        denom[denom == 0] = 1e-12
        R /= denom
    elif method == "roth":
        denom = np.abs(SIG2)**2
        denom[denom == 0] = 1e-12
        R /= denom
    elif method == "scot":
        denom = np.sqrt(np.abs(SIG1)**2 * np.abs(SIG2)**2)
        denom[denom == 0] = 1e-12
        R /= denom
    elif method == "ml":
        # Implementación de ML aquí
        # Por ahora, solo devuelve nan
        return np.nan, time.perf_counter() - start_time
    # Otros métodos pueden agregarse aquí
    cc = np.fft.ifft(R).real
    lags = correlation_lags(len(sig1), len(sig2), mode='full')
    tdoa_index = np.argmax(cc)
    tdoa = lags[tdoa_index] / fs
    return tdoa, time.perf_counter() - start_time



"""
preprocess.py
-------------
Clean and detrend a raw light curve before transit searching.

Steps:
  1. sigma-clip outliers
  2. flatten slow stellar/instrumental variability with a Savitzky-Golay
     running filter, while protecting in-transit points from being erased
  3. return normalised flux (median = 1.0)
"""
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except ImportError:
    _HAS_SAVGOL = False


def _filter(y, win, polyorder):
    if _HAS_SAVGOL:
        return savgol_filter(y, win, polyorder)
    half = max(1, win // 2)
    padded = np.pad(y, half, mode='edge')
    return np.array([np.median(padded[i:i + win]) for i in range(len(y))])


def sigma_clip(t, flux, err, sigma=5, iters=3):
    mask = np.ones_like(flux, dtype=bool)
    for _ in range(iters):
        med = np.median(flux[mask])
        std = np.std(flux[mask])
        new = np.abs(flux - med) < sigma * std
        if new.sum() == mask.sum():
            break
        mask = new
    return t[mask], flux[mask], err[mask]


def flatten(t, flux, window_frac=0.05, polyorder=2, protect_sigma=3.0):
    """
    Detrend using Savitzky-Golay (or moving median fallback). Dips (potential transits)
    are masked out when estimating the trend so they survive the flattening.
    """
    n = len(flux)
    win = max(11, int(window_frac * n) | 1)   # odd window
    if win >= n:
        win = (n // 2) * 2 - 1
    win = max(win, polyorder + 2)
    if win % 2 == 0:
        win += 1

    # first pass trend
    trend = _filter(flux, win, polyorder)
    resid = flux - trend
    scatter = np.std(resid)
    # mask points well below the trend (transits/eclipses are dips)
    keep = resid > -protect_sigma * scatter

    # re-fit trend using only out-of-dip points (interpolate over masked)
    trend2 = np.interp(t, t[keep], _filter(flux[keep], 
              min(win, (keep.sum()//2)*2-1 if keep.sum() > win else win), polyorder)) \
              if keep.sum() > win else trend

    flat = flux / trend2
    flat = flat / np.median(flat)
    return flat, trend2


def preprocess(t, flux, err, window_frac=0.05):
    """Full clean+flatten. Returns (t, flat_flux, err)."""
    t, flux, err = sigma_clip(t, flux, err)
    flat, trend = flatten(t, flux, window_frac=window_frac)
    # propagate errors (approx: divide by trend)
    err = err / trend
    return t, flat, err

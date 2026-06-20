"""
search.py
---------
Box Least Squares (BLS) periodic transit search.

Finds the best-fit period, epoch, duration and depth of a box-shaped dip,
and reports detection statistics (BLS power, depth SNR).
"""
import numpy as np
from astropy.timeseries import BoxLeastSquares


def run_bls(t, flux, err, period_min=0.5, period_max=None,
            n_periods=3000, n_durations=6):
    """
    Search for the strongest periodic box dip.

    Returns a dict with period, t0, duration, depth, snr, power, and the
    full BLS periodogram for plotting / vetting.
    """
    baseline = t.max() - t.min()
    if period_max is None:
        period_max = max(period_min + 1.0, baseline / 2.0)

    bls = BoxLeastSquares(t, flux, dy=err)
    periods = np.linspace(period_min, period_max, n_periods)
    durations = np.linspace(0.03, 0.30, n_durations)
    power = bls.power(periods, durations)

    i = int(np.argmax(power.power))
    res = dict(
        period=float(power.period[i]),
        t0=float(power.transit_time[i]),
        duration=float(power.duration[i]),
        depth=float(power.depth[i]),
        snr=float(power.depth_snr[i]),
        power=float(power.power[i]),
        log_likelihood=float(power.log_likelihood[i]) if hasattr(power, "log_likelihood") else np.nan,
        periodogram=dict(period=np.asarray(power.period),
                         power=np.asarray(power.power)),
    )
    # robust SNR: depth / (scatter / sqrt(N_in_transit))
    res["snr_robust"] = _transit_snr(t, flux, res)
    return res


def _transit_snr(t, flux, res):
    p, t0, dur, depth = res["period"], res["t0"], res["duration"], res["depth"]
    phase = ((t - t0 + 0.5 * p) % p) - 0.5 * p
    in_tr = np.abs(phase) < dur / 2
    oot = ~in_tr
    if in_tr.sum() < 3 or oot.sum() < 10:
        return 0.0
    scatter = np.std(flux[oot])
    n_in = in_tr.sum()
    return float(abs(depth) / (scatter / np.sqrt(n_in))) if scatter > 0 else 0.0


def has_significant_dip(res, power_threshold=10.0, snr_threshold=7.0):
    """Quick gate: is there a periodic dip worth classifying?"""
    return (res["snr_robust"] >= snr_threshold) or (res["power"] >= power_threshold)

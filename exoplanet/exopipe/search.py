"""
search.py
---------
Box Least Squares (BLS) periodic transit search.

Finds the best-fit period, epoch, duration and depth of a box-shaped dip,
and reports detection statistics (BLS power, depth SNR).
"""
import numpy as np

try:
    from astropy.timeseries import BoxLeastSquares
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False


class _FallbackPowerResult:
    def __init__(self, period, power, transit_time, duration, depth, depth_snr):
        self.period = np.asarray(period)
        self.power = np.asarray(power)
        self.transit_time = np.asarray(transit_time)
        self.duration = np.asarray(duration)
        self.depth = np.asarray(depth)
        self.depth_snr = np.asarray(depth_snr)


class _FallbackBoxLeastSquares:
    def __init__(self, t, flux, dy=None):
        self.t = np.asarray(t)
        self.flux = np.asarray(flux)
        self.dy = np.asarray(dy) if dy is not None else np.ones_like(self.flux)

    def power(self, periods, durations):
        powers = []
        t0s = []
        depths = []
        best_durs = []
        snrs = []

        for p in periods:
            best_p_power = -1.0
            best_p_t0 = float(self.t.min())
            best_p_dur = float(durations[0])
            best_p_depth = 0.0
            best_p_snr = 0.0

            n_bins = 20
            phase = ((self.t - self.t.min()) % p) / p
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_idx = np.digitize(phase, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            bin_fluxes = np.array([
                np.median(self.flux[bin_idx == b]) if np.any(bin_idx == b) else 1.0
                for b in range(n_bins)
            ])
            min_bin = int(np.argmin(bin_fluxes))
            t0_candidate = float(self.t.min() + (min_bin / n_bins) * p)

            for dur in durations:
                dur_phase = dur / p
                in_transit = np.abs(((phase - (min_bin / n_bins) + 0.5) % 1.0) - 0.5) < (dur_phase / 2.0)
                n_in = int(np.sum(in_transit))
                n_out = len(self.flux) - n_in

                if n_in >= 2 and n_out >= 5:
                    f_in = float(np.mean(self.flux[in_transit]))
                    f_out = float(np.mean(self.flux[~in_transit]))
                    d = f_out - f_in
                    std_out = float(np.std(self.flux[~in_transit])) + 1e-9
                    snr = (d / std_out) * np.sqrt(n_in)
                    pow_val = max(0.0, float(snr))

                    if pow_val > best_p_power:
                        best_p_power = pow_val
                        best_p_t0 = t0_candidate
                        best_p_dur = float(dur)
                        best_p_depth = max(0.0, float(d))
                        best_p_snr = float(snr)

            powers.append(best_p_power if best_p_power > 0 else 0.0)
            t0s.append(best_p_t0)
            depths.append(best_p_depth)
            best_durs.append(best_p_dur)
            snrs.append(best_p_snr)

        return _FallbackPowerResult(
            period=periods,
            power=powers,
            transit_time=t0s,
            duration=best_durs,
            depth=depths,
            depth_snr=snrs
        )


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

    if _HAS_ASTROPY:
        bls = BoxLeastSquares(t, flux, dy=err)
    else:
        # Optimize step count for pure numpy fallback
        n_periods = min(n_periods, 400)
        n_durations = min(n_durations, 4)
        bls = _FallbackBoxLeastSquares(t, flux, dy=err)
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

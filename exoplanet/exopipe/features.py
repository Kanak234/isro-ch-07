"""
features.py
-----------
Extract physically-motivated features that separate the four classes:

  transit : shallow, U-shaped, NO secondary, equal odd/even depths
  eclipse : deep, secondary eclipse present, often unequal odd/even, V-shaped
  blend   : very shallow + extra correlated noise, sometimes odd shape
  other   : no coherent periodic dip (low BLS SNR, high residual scatter)

These features feed the gradient-boosted classifier (classify.py).
"""
import numpy as np


def _fold(t, flux, period, t0):
    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def _binned(phase, flux, nbins=200):
    edges = np.linspace(-0.5, 0.5, nbins + 1)
    idx = np.digitize(phase, edges) - 1
    idx = np.clip(idx, 0, nbins - 1)
    out = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            out[b] = np.median(flux[m])
    # fill gaps
    good = np.isfinite(out)
    if good.sum() > 2:
        out = np.interp(np.arange(nbins), np.arange(nbins)[good], out[good])
    return out


def _depth_near(phase, flux, centre, half):
    m = np.abs(phase - centre) < half
    if m.sum() < 3:
        return 0.0
    return float(1.0 - np.median(flux[m]))


def extract(t, flux, err, res):
    """Return a feature vector (and a dict of named features for inspection)."""
    p, t0, dur = res["period"], res["t0"], res["duration"]
    half = (dur / p) / 2.0
    phase, fsorted = _fold(t, flux, p, t0)

    # --- depth & shape near primary (phase 0) ---
    primary_depth = _depth_near(phase, fsorted, 0.0, half)

    # secondary eclipse near phase +0.5
    secondary_depth = _depth_near(phase, fsorted, 0.5, half)
    sec_ratio = secondary_depth / (primary_depth + 1e-9)

    # --- odd / even depth difference (EB signature) ---
    phase2, f2 = _fold(t, flux, 2 * p, t0)
    odd_depth = _depth_near(phase2, f2, 0.0, half / 2)
    even_depth = _depth_near(phase2, f2, 0.5, half / 2)
    odd_even = abs(odd_depth - even_depth) / (odd_depth + even_depth + 1e-9)

    # --- transit shape: U vs V (curvature of the floor) ---
    in_tr = np.abs(phase) < half
    if in_tr.sum() > 5:
        x = phase[in_tr]
        y = fsorted[in_tr]
        # fit parabola; U-shape (planet) => small |a| relative to depth,
        # V-shape (grazing/EB) => sharp -> high curvature ratio
        try:
            a, b, c = np.polyfit(x, y, 2)
            v_score = float(abs(a) * half**2 / (primary_depth + 1e-9))
        except Exception:
            v_score = 0.0
        floor_flatness = float(np.std(y))
    else:
        v_score, floor_flatness = 0.0, 0.0

    # --- noise / scatter outside transit ---
    oot = ~in_tr
    oot_scatter = float(np.std(fsorted[oot])) if oot.sum() > 5 else 0.0
    # ratio of dip depth to out-of-transit scatter (blend => low)
    depth_to_noise = primary_depth / (oot_scatter + 1e-9)

    # --- duration / period ratio (physical plausibility) ---
    dur_frac = dur / p

    feats = {
        "bls_depth": res["depth"],
        "bls_snr": res["snr_robust"],
        "bls_power": res["power"],
        "primary_depth": primary_depth,
        "secondary_depth": secondary_depth,
        "secondary_ratio": sec_ratio,
        "odd_even_diff": odd_even,
        "v_score": v_score,
        "floor_flatness": floor_flatness,
        "oot_scatter": oot_scatter,
        "depth_to_noise": depth_to_noise,
        "duration_frac": dur_frac,
        "period": p,
    }
    vec = np.array([
        feats["bls_depth"], feats["bls_snr"], feats["bls_power"],
        feats["primary_depth"], feats["secondary_depth"], feats["secondary_ratio"],
        feats["odd_even_diff"], feats["v_score"], feats["floor_flatness"],
        feats["oot_scatter"], feats["depth_to_noise"], feats["duration_frac"],
    ], dtype=float)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, feats


FEATURE_NAMES = [
    "bls_depth", "bls_snr", "bls_power", "primary_depth", "secondary_depth",
    "secondary_ratio", "odd_even_diff", "v_score", "floor_flatness",
    "oot_scatter", "depth_to_noise", "duration_frac",
]

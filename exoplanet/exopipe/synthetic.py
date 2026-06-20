"""
synthetic.py
------------
Generate realistic synthetic TESS-like light curves for four classes:
  - transit  : planetary transit (U-shaped, no secondary, equal odd/even)
  - eclipse  : eclipsing binary (V-shaped or deep, secondary eclipse, odd/even diff)
  - blend    : diluted transit/eclipse from a neighbouring source (shallow, noisy)
  - other    : starspots / stellar variability / pure noise (no coherent periodic dip)

These are used to (a) build a labelled training set for the classifier and
(b) end-to-end test the pipeline when the MAST archive is unreachable.

When you have network access, replace synthetic curves with real TESS light
curves via exopipe.data.download_tess() -- the rest of the pipeline is identical.
"""
import numpy as np

try:
    import batman
    _HAS_BATMAN = True
except Exception:
    _HAS_BATMAN = False


TESS_CADENCE_DAYS = 2.0 / (60 * 24)   # 2-minute short cadence
SECTOR_LENGTH = 27.0                  # days in a TESS sector


def _time_grid(length=SECTOR_LENGTH, cadence=TESS_CADENCE_DAYS, gap_frac=0.04):
    """Time array with a realistic mid-sector data-downlink gap."""
    t = np.arange(0, length, cadence)
    # remove a contiguous gap near the middle (downlink)
    g0 = length * 0.5
    mask = ~((t > g0) & (t < g0 + length * gap_frac))
    return t[mask]


def _batman_transit(t, t0, per, rp, a, inc, u=(0.3, 0.2)):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rp          # Rp/Rs
    params.a = a            # a/Rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(u)
    params.limb_dark = "quadratic"
    m = batman.TransitModel(params, t)
    return m.light_curve(params)


def _box_dip(t, t0, per, depth, dur, phase_offset=0.0):
    """Fallback box-shaped dip if batman is unavailable."""
    phase = ((t - t0 - phase_offset) % per) / per
    phase[phase > 0.5] -= 1.0
    half = (dur / per) / 2.0
    flux = np.ones_like(t)
    flux[np.abs(phase) < half] -= depth
    return flux


def make_transit(noise=4e-4, seed=None):
    rng = np.random.default_rng(seed)
    t = _time_grid()
    per = rng.uniform(2.0, 12.0)
    t0 = rng.uniform(0, per)
    rp = rng.uniform(0.04, 0.11)          # planet: Rp/Rs small -> shallow, U-shaped
    a = rng.uniform(8, 22)
    inc = rng.uniform(87.5, 90.0)
    if _HAS_BATMAN:
        flux = _batman_transit(t, t0, per, rp, a, inc)
    else:
        flux = _box_dip(t, t0, per, rp**2, 0.1 * per)
    flux = _add_systematics(t, flux, rng)
    flux += rng.normal(0, noise, len(t))
    meta = dict(label="transit", period=per, t0=t0, depth=rp**2,
                duration=_dur_from_geometry(per, a, inc, rp), rp=rp)
    return t, flux, noise * np.ones_like(t), meta


def make_eclipse(noise=6e-4, seed=None):
    """Eclipsing binary: deep, secondary eclipse, possible odd/even difference."""
    rng = np.random.default_rng(seed)
    t = _time_grid()
    per = rng.uniform(1.0, 10.0)
    t0 = rng.uniform(0, per)
    depth_pri = rng.uniform(0.02, 0.15)       # deep primary
    depth_sec = depth_pri * rng.uniform(0.2, 0.8)   # secondary eclipse present
    dur = rng.uniform(0.05, 0.12) * per
    flux = np.ones_like(t)
    # primary
    flux *= _v_shaped(t, t0, per, depth_pri, dur)
    # secondary at phase 0.5
    flux *= _v_shaped(t, t0 + per / 2, per, depth_sec, dur)
    # occasional odd/even depth difference (heartbeat / unequal stars)
    if rng.random() < 0.5:
        oe = rng.uniform(0.7, 0.95)
        flux *= _v_shaped(t, t0, 2 * per, depth_pri * (oe - 1), dur)
    flux = _add_systematics(t, flux, rng)
    flux += rng.normal(0, noise, len(t))
    meta = dict(label="eclipse", period=per, t0=t0, depth=depth_pri,
                duration=dur, secondary=depth_sec)
    return t, flux, noise * np.ones_like(t), meta


def make_blend(noise=8e-4, seed=None):
    """Diluted signal from a neighbour in a crowded aperture: shallow + extra noise."""
    rng = np.random.default_rng(seed)
    t = _time_grid()
    per = rng.uniform(1.0, 12.0)
    t0 = rng.uniform(0, per)
    # underlying deep eclipse, diluted by a bright contaminant
    true_depth = rng.uniform(0.02, 0.12)
    dilution = rng.uniform(0.05, 0.25)         # only a fraction survives
    obs_depth = true_depth * dilution
    dur = rng.uniform(0.04, 0.10) * per
    flux = _v_shaped(t, t0, per, obs_depth, dur)
    # blends often show a centroid wobble -> mimic via correlated extra noise
    flux = _add_systematics(t, flux, rng, strength=1.8)
    flux += rng.normal(0, noise, len(t))
    flux += 0.5 * noise * np.sin(2 * np.pi * t / rng.uniform(0.3, 0.8))
    meta = dict(label="blend", period=per, t0=t0, depth=obs_depth,
                duration=dur, dilution=dilution)
    return t, flux, noise * np.ones_like(t), meta


def make_other(noise=5e-4, seed=None):
    """Starspots / pulsation / pure noise -- no coherent transit-like dip."""
    rng = np.random.default_rng(seed)
    t = _time_grid()
    flux = np.ones_like(t)
    kind = rng.choice(["spots", "pulsation", "noise"])
    if kind == "spots":
        prot = rng.uniform(3, 15)
        flux += rng.uniform(0.005, 0.03) * np.sin(2 * np.pi * t / prot + rng.uniform(0, 6))
        flux += 0.4 * rng.uniform(0.005, 0.03) * np.sin(4 * np.pi * t / prot)
    elif kind == "pulsation":
        for _ in range(rng.integers(2, 5)):
            f = rng.uniform(0.5, 8)
            flux += rng.uniform(0.001, 0.01) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6))
    flux = _add_systematics(t, flux, rng)
    flux += rng.normal(0, noise, len(t))
    meta = dict(label="other", period=np.nan, t0=np.nan, depth=np.nan,
                duration=np.nan, kind=kind)
    return t, flux, noise * np.ones_like(t), meta


# ---------- helpers ----------

def _v_shaped(t, t0, per, depth, dur):
    """A V-shaped (grazing/EB-like) dip, multiplicative (returns flux factor)."""
    phase = ((t - t0) % per) / per
    phase[phase > 0.5] -= 1.0
    half = (dur / per) / 2.0
    flux = np.ones_like(t)
    inside = np.abs(phase) < half
    # linear V: deepest at centre, zero at edges
    flux[inside] -= depth * (1 - np.abs(phase[inside]) / half)
    return flux


def _add_systematics(t, flux, rng, strength=1.0):
    """Slow instrumental trend (focus/thermal) common to all TESS curves."""
    trend = (strength * rng.uniform(1e-4, 8e-4) * (t - t.mean()) +
             strength * rng.uniform(1e-4, 5e-4) * np.sin(2 * np.pi * t / rng.uniform(8, 20)))
    return flux + trend


def _dur_from_geometry(per, a, inc, rp):
    b = a * np.cos(np.radians(inc))
    arg = ((1 + rp) ** 2 - b ** 2)
    if arg <= 0:
        return 0.05 * per
    return (per / np.pi) * np.arcsin(np.sqrt(arg) / a)


GENERATORS = {
    "transit": make_transit,
    "eclipse": make_eclipse,
    "blend": make_blend,
    "other": make_other,
}


def make_dataset(n_per_class=150, seed=0):
    """Build a labelled training set: returns list of (t, flux, err, meta)."""
    rng = np.random.default_rng(seed)
    curves = []
    for label, gen in GENERATORS.items():
        for _ in range(n_per_class):
            s = int(rng.integers(0, 2**31))
            curves.append(gen(seed=s))
    rng.shuffle(curves)
    return curves

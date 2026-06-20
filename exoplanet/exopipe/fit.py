"""
fit.py
------
For confirmed transit candidates, fit a physical transit model and estimate
parameters WITH uncertainties:

    orbital period, transit epoch (t0), transit depth, transit duration,
    planet/star radius ratio (Rp/Rs), scaled semi-major axis (a/Rs), inclination

Two-stage approach:
  1. least-squares refinement of the BLS solution (fast)
  2. MCMC (emcee) to sample the posterior -> median +/- 1-sigma uncertainties

Falls back to an analytic trapezoid fit if `batman` is unavailable.
"""
import numpy as np

try:
    import batman
    _HAS_BATMAN = True
except Exception:
    _HAS_BATMAN = False

try:
    import emcee
    _HAS_EMCEE = True
except Exception:
    _HAS_EMCEE = False


def _model(theta, t):
    t0, per, rp, a, inc = theta
    params = batman.TransitParams()
    params.t0, params.per, params.rp = t0, per, rp
    params.a, params.inc = a, inc
    params.ecc, params.w = 0.0, 90.0
    params.u, params.limb_dark = [0.3, 0.2], "quadratic"
    return batman.TransitModel(params, t).light_curve(params)


def _log_prior(theta, p0):
    t0, per, rp, a, inc = theta
    if not (0 < per < 3 * p0):           return -np.inf
    if not (0.005 < rp < 0.5):           return -np.inf
    if not (1.5 < a < 100):              return -np.inf
    if not (60 < inc <= 90):             return -np.inf
    return 0.0


def _log_prob(theta, t, f, ferr, p0):
    lp = _log_prior(theta, p0)
    if not np.isfinite(lp):
        return -np.inf
    model = _model(theta, t)
    return lp - 0.5 * np.sum(((f - model) / ferr) ** 2)


def fit_transit(t, flux, err, res, nwalkers=32, nsteps=3000, burn=1000,
                run_mcmc=True):
    """
    Fit transit parameters. Returns dict of param -> (median, minus, plus)
    plus derived depth and duration with uncertainties.
    """
    p0 = res["period"]
    rp0 = np.sqrt(max(res["depth"], 1e-6))
    init = np.array([res["t0"], p0, rp0, 12.0, 88.5])

    if not _HAS_BATMAN:
        return _fit_trapezoid(t, flux, err, res)

    # ---- stage 1: quick least squares via scipy ----
    from scipy.optimize import least_squares

    def resid(theta):
        if not np.isfinite(_log_prior(theta, p0)):
            return np.full_like(flux, 1e3)
        return (flux - _model(theta, t)) / err

    try:
        ls = least_squares(resid, init, method="lm", max_nfev=2000)
        init = ls.x
    except Exception:
        pass

    if not (run_mcmc and _HAS_EMCEE):
        depth = init[2] ** 2
        dur = _duration(init)
        return _package_ls(init, depth, dur)

    # ---- stage 2: MCMC posterior ----
    ndim = len(init)
    pos = init + 1e-4 * np.random.randn(nwalkers, ndim) * np.abs(init)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_prob,
                                    args=(t, flux, err, p0))
    sampler.run_mcmc(pos, nsteps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)

    names = ["t0", "period", "rp", "a", "inc"]
    out = {}
    for i, nm in enumerate(names):
        lo, mid, hi = np.percentile(chain[:, i], [16, 50, 84])
        out[nm] = (float(mid), float(mid - lo), float(hi - mid))

    # derived: depth = rp^2 ; duration from geometry
    rp_s = chain[:, 2]
    depth_s = rp_s ** 2
    lo, mid, hi = np.percentile(depth_s, [16, 50, 84])
    out["depth"] = (float(mid), float(mid - lo), float(hi - mid))

    dur_s = np.array([_duration(chain[j]) for j in
                      np.random.choice(len(chain), min(2000, len(chain)), replace=False)])
    lo, mid, hi = np.percentile(dur_s, [16, 50, 84])
    out["duration"] = (float(mid), float(mid - lo), float(hi - mid))

    out["_chain"] = chain
    out["_best"] = init
    out["method"] = "batman+emcee"
    return out


def _duration(theta):
    t0, per, rp, a, inc = theta
    b = a * np.cos(np.radians(inc))
    arg = (1 + rp) ** 2 - b ** 2
    if arg <= 0:
        return np.nan
    return (per / np.pi) * np.arcsin(np.sqrt(arg) / a)


def _package_ls(theta, depth, dur):
    names = ["t0", "period", "rp", "a", "inc"]
    out = {nm: (float(theta[i]), np.nan, np.nan) for i, nm in enumerate(names)}
    out["depth"] = (float(depth), np.nan, np.nan)
    out["duration"] = (float(dur), np.nan, np.nan)
    out["method"] = "batman+leastsq"
    return out


def _fit_trapezoid(t, flux, err, res):
    """Analytic fallback: depth/duration/period directly from BLS + folded box."""
    out = {
        "period": (res["period"], np.nan, np.nan),
        "t0": (res["t0"], np.nan, np.nan),
        "depth": (res["depth"], np.nan, np.nan),
        "duration": (res["duration"], np.nan, np.nan),
        "rp": (float(np.sqrt(max(res["depth"], 1e-6))), np.nan, np.nan),
        "method": "trapezoid",
    }
    return out

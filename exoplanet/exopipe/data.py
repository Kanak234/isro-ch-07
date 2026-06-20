"""
data.py
-------
Download and load REAL TESS light curves from the MAST archive.

Use this on a machine WITH internet access. The output format
(time, flux, flux_err arrays) is identical to exopipe.synthetic, so the
rest of the pipeline (preprocess -> BLS -> features -> classify -> fit)
works unchanged whether the data is real or synthetic.

Data source: https://archive.stsci.edu/tess/  (via the lightkurve package)
"""
import numpy as np

try:
    import lightkurve as lk
    _HAS_LK = True
except Exception:
    _HAS_LK = False


def download_tess(tic_id, sector=None, author="SPOC", cadence="short"):
    """
    Download a TESS light curve for a given TIC ID.

    Parameters
    ----------
    tic_id : int or str   e.g. 307210830
    sector : int or None  specific TESS sector (None = all available, stitched)
    author : str          'SPOC' (2-min) or 'TESS-SPOC' / 'QLP' etc.
    cadence: str          'short' (2-min, recommended) or 'long'

    Returns
    -------
    t, flux, flux_err : np.ndarray   (normalised flux, NaNs/outliers removed)
    meta : dict
    """
    if not _HAS_LK:
        raise RuntimeError("lightkurve not installed. pip install lightkurve")

    sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS",
                              author=author, sector=sector, cadence=cadence)
    if len(sr) == 0:
        raise ValueError(f"No light curves found for TIC {tic_id}")

    lc = sr.download_all().stitch()
    lc = lc.remove_nans().remove_outliers(sigma=5)

    t = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    err = np.asarray(lc.flux_err.value, dtype=float)

    # normalise to median 1.0
    med = np.nanmedian(flux)
    flux = flux / med
    err = err / med

    good = np.isfinite(t) & np.isfinite(flux) & np.isfinite(err)
    meta = dict(label="unknown", tic=str(tic_id), sector=sector, source=author)
    return t[good], flux[good], err[good], meta


def download_sector_targets(sector, limit=None, author="SPOC"):
    """
    Yield light curves for many targets in a TESS sector.
    Useful for scanning the ~20-30k high-cadence curves the brief mentions.

    NOTE: a full sector is large; use `limit` while developing.
    """
    if not _HAS_LK:
        raise RuntimeError("lightkurve not installed.")
    sr = lk.search_lightcurve(f"sector {sector}", mission="TESS",
                              author=author, cadence="short")
    n = len(sr) if limit is None else min(limit, len(sr))
    for i in range(n):
        try:
            lc = sr[i].download().remove_nans().remove_outliers(sigma=5)
            med = np.nanmedian(lc.flux.value)
            yield (np.asarray(lc.time.value, float),
                   np.asarray(lc.flux.value, float) / med,
                   np.asarray(lc.flux_err.value, float) / med,
                   dict(label="unknown", index=i))
        except Exception as e:
            print(f"  [skip {i}] {type(e).__name__}: {e}")

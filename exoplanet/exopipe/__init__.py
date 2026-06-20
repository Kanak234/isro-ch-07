"""
exopipe : AI-enabled detection & classification of exoplanet transits
         from noisy TESS light curves.
"""
from . import synthetic, preprocess, search, features, classify, fit, visualize, ensemble
try:
    from . import data
except Exception:
    data = None
try:
    from . import cnn
except Exception:
    cnn = None

__all__ = ["synthetic", "data", "preprocess", "search",
           "features", "classify", "cnn", "ensemble", "fit", "visualize"]
__version__ = "1.1"

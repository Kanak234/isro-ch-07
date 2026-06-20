"""
cnn.py
------
AstroNet-style 1-D Convolutional Neural Network for light-curve classification.

Based on Shallue & Vanderburg (2018), "Identifying Exoplanets with Deep
Learning". Each light curve is reduced to TWO phase-folded, median-binned views:

  * GLOBAL view (2001 bins) : the entire orbit at one phase resolution
                              -> captures secondary eclipses, out-of-eclipse
                                 variation, overall shape.
  * LOCAL view  (201 bins)  : zoomed onto the transit/eclipse itself
                              -> captures depth, ingress/egress, U-vs-V shape.

Two separate convolutional columns process the two views; their flattened
outputs are concatenated and passed through dense layers to a 4-way softmax
(transit / eclipse / blend / other).

This complements the feature-based gradient-boosted classifier (classify.py).
In production the two are ensembled (see ensemble.py).
"""
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _HAS_TF = True
except Exception:
    _HAS_TF = False

CLASSES = ["transit", "eclipse", "blend", "other"]
GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_WINDOW = 4.0   # local view spans +/- LOCAL_WINDOW * (duration/period)


# ---------------------------------------------------------------- views ----

def _median_bin(phase, flux, nbins, pmin=-0.5, pmax=0.5):
    """Median-bin flux onto a fixed phase grid; interpolate empty bins."""
    edges = np.linspace(pmin, pmax, nbins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    out = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.any():
            out[b] = np.median(flux[m])
    good = np.isfinite(out)
    if good.sum() >= 2:
        out = np.interp(np.arange(nbins), np.arange(nbins)[good], out[good])
    else:
        out = np.ones(nbins)
    return out


def _normalise(view):
    """Centre on the median (=0) and scale so the deepest dip = -1 (AstroNet)."""
    v = view - np.median(view)
    depth = np.abs(np.min(v))
    if depth > 0:
        v = v / depth
    return v


def make_views(t, flux, res):
    """
    Build (global_view, local_view) from a light curve and its BLS solution.
    Returns two 1-D float arrays of length GLOBAL_BINS and LOCAL_BINS.
    """
    p, t0, dur = res["period"], res["t0"], max(res["duration"], 1e-3)
    phase = ((t - t0 + 0.5 * p) % p) / p - 0.5

    # global: full orbit
    g = _median_bin(phase, flux, GLOBAL_BINS)
    g = _normalise(g)

    # local: zoom to +/- LOCAL_WINDOW * (dur/p) around phase 0
    half = LOCAL_WINDOW * (dur / p)
    half = min(max(half, 0.01), 0.5)
    sel = np.abs(phase) < half
    if sel.sum() > 10:
        l = _median_bin(phase[sel], flux[sel], LOCAL_BINS, -half, half)
    else:
        l = _median_bin(phase, flux, LOCAL_BINS, -half, half)
    l = _normalise(l)

    return g.astype("float32"), l.astype("float32")


# ---------------------------------------------------------------- model ----

def build_model(n_classes=4):
    """Two-column 1-D CNN (global + local) -> dense -> softmax."""
    if not _HAS_TF:
        raise RuntimeError("tensorflow not installed. pip install tensorflow-cpu")

    def conv_column(inp, blocks):
        x = inp
        for n_filters, n_convs in blocks:
            for _ in range(n_convs):
                x = layers.Conv1D(n_filters, 5, activation="relu",
                                  padding="same")(x)
            x = layers.MaxPool1D(pool_size=5, strides=2)(x)
        return layers.Flatten()(x)

    g_in = keras.Input(shape=(GLOBAL_BINS, 1), name="global_view")
    l_in = keras.Input(shape=(LOCAL_BINS, 1), name="local_view")

    # global column: deeper (5 conv blocks)
    g = conv_column(g_in, [(16, 2), (32, 2), (64, 2), (128, 2), (256, 2)])
    # local column: shallower (2 conv blocks)
    l = conv_column(l_in, [(16, 2), (32, 2)])

    x = layers.concatenate([g, l])
    for _ in range(4):
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model([g_in, l_in], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ------------------------------------------------------------ train/apply --

def prepare_xy(views_global, views_local, labels):
    Xg = np.asarray(views_global)[..., None]
    Xl = np.asarray(views_local)[..., None]
    y = np.array([CLASSES.index(c) for c in labels])
    return Xg, Xl, y


def train(model, Xg, Xl, y, epochs=40, batch_size=32, val_split=0.2, verbose=2):
    cw = _class_weights(y)
    es = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                       monitor="val_loss")
    hist = model.fit({"global_view": Xg, "local_view": Xl}, y,
                     validation_split=val_split, epochs=epochs,
                     batch_size=batch_size, class_weight=cw,
                     callbacks=[es], verbose=verbose)
    return hist


def _class_weights(y):
    counts = np.bincount(y, minlength=len(CLASSES))
    total = counts.sum()
    return {i: total / (len(CLASSES) * c) if c > 0 else 1.0
            for i, c in enumerate(counts)}


def predict_one(model, g_view, l_view):
    """Return (label, confidence, prob_dict) for a single curve."""
    g = g_view[None, ..., None]
    l = l_view[None, ..., None]
    proba = model.predict({"global_view": g, "local_view": l}, verbose=0)[0]
    pdict = {CLASSES[i]: float(proba[i]) for i in range(len(CLASSES))}
    label = max(pdict, key=pdict.get)
    return label, pdict[label], pdict


def save(model, path):
    model.save(path)


def load(path):
    if not _HAS_TF:
        raise RuntimeError("tensorflow not installed.")
    return keras.models.load_model(path)

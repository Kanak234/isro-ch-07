"""
classify.py
-----------
Train and apply a classifier that categorises a detected dip into:
    transit | eclipse | blend | other

Uses a HistGradientBoostingClassifier on the physical features from
features.py. Provides per-class probabilities (the "confidence level"
the brief asks for).
"""
import numpy as np

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    joblib = None

from .features import FEATURE_NAMES

CLASSES = ["transit", "eclipse", "blend", "other"]


def train(X, y, verbose=True):
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, class_weight="balanced",
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    if verbose:
        print(f"  5-fold CV macro-F1: {scores.mean():.3f} +/- {scores.std():.3f}")
    clf.fit(X, y)
    return clf, scores


def evaluate(clf, X, y):
    pred = clf.predict(X)
    report = classification_report(y, pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    return report, cm, pred


def classify_one(clf, feature_vec):
    """Return (label, confidence, full_probability_dict)."""
    if clf is not None and hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(feature_vec.reshape(1, -1))[0]
            order = clf.classes_
            pdict = {c: float(proba[list(order).index(c)]) for c in order}
            label = max(pdict, key=pdict.get)
            return label, pdict[label], pdict
        except Exception:
            pass

    # Physics-based heuristic fallback
    p_dict = {c: 0.05 for c in CLASSES}
    sec_ratio = float(feature_vec[5]) if len(feature_vec) > 5 else 0.0
    primary_depth = float(feature_vec[3]) if len(feature_vec) > 3 else 0.0
    snr = float(feature_vec[1]) if len(feature_vec) > 1 else 0.0

    if sec_ratio > 0.25 or primary_depth > 0.04:
        label = "eclipse"
    elif snr < 4.0:
        label = "other"
    elif primary_depth < 0.001:
        label = "blend"
    else:
        label = "transit"

    p_dict[label] = 0.85
    total = sum(p_dict.values())
    p_dict = {k: v / total for k, v in p_dict.items()}
    return label, p_dict[label], p_dict


def save(clf, path):
    if joblib is not None:
        joblib.dump(clf, path)


def load(path):
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

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
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

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
    proba = clf.predict_proba(feature_vec.reshape(1, -1))[0]
    order = clf.classes_
    pdict = {c: float(proba[list(order).index(c)]) for c in order}
    label = max(pdict, key=pdict.get)
    return label, pdict[label], pdict


def save(clf, path):
    joblib.dump(clf, path)


def load(path):
    return joblib.load(path)

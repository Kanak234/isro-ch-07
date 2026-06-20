"""
ensemble.py
-----------
Combine the two complementary classifiers:

  1. gradient-boosted trees on physical features (classify.py)  -- robust,
     interpretable, strong on engineered physics.
  2. AstroNet-style 1-D CNN on phase-folded views (cnn.py)      -- learns
     subtle shape information directly from the light curve.

The ensemble averages their per-class probabilities (optionally weighted).
This is how production vetting pipelines (e.g. ExoMiner, AstroNet-Vetting)
gain robustness: the two models fail on different cases.
"""
import numpy as np

CLASSES = ["transit", "eclipse", "blend", "other"]


def combine_proba(p_tree, p_cnn, w_tree=0.5, w_cnn=0.5):
    """
    Average two probability dicts. Returns (label, confidence, prob_dict).
    Missing model -> the other one is used alone.
    """
    if p_tree is None and p_cnn is None:
        raise ValueError("at least one model's probabilities required")
    if p_cnn is None:
        merged = dict(p_tree)
    elif p_tree is None:
        merged = dict(p_cnn)
    else:
        s = w_tree + w_cnn
        merged = {c: (w_tree * p_tree.get(c, 0) + w_cnn * p_cnn.get(c, 0)) / s
                  for c in CLASSES}
    # renormalise
    tot = sum(merged.values()) or 1.0
    merged = {c: v / tot for c, v in merged.items()}
    label = max(merged, key=merged.get)
    return label, merged[label], merged


def classify_ensemble(feature_vec, g_view, l_view,
                      tree_clf=None, cnn_model=None,
                      w_tree=0.5, w_cnn=0.5):
    """
    Run whichever models are supplied and return the combined verdict.
    """
    p_tree = p_cnn = None
    if tree_clf is not None:
        from .classify import classify_one
        _, _, p_tree = classify_one(tree_clf, feature_vec)
    if cnn_model is not None:
        from .cnn import predict_one
        _, _, p_cnn = predict_one(cnn_model, g_view, l_view)
    return combine_proba(p_tree, p_cnn, w_tree, w_cnn)

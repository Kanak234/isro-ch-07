#!/usr/bin/env python3
"""
train_classifier.py
-------------------
Build a labelled feature table from training light curves, train the
classifier, evaluate it, and save the model.

By default uses synthetic curves (synthetic.make_dataset). To train on a
real curated set, load your labelled curves into the `curves` list as
(t, flux, err, meta) tuples where meta['label'] is one of
transit/eclipse/blend/other, then run the same pipeline.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe import synthetic, preprocess, search, features, classify, visualize

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def build_feature_table(curves):
    X, y = [], []
    skipped = 0
    for i, (t, flux, err, meta) in enumerate(curves):
        try:
            tt, ff, ee = preprocess.preprocess(t, flux, err)
            res = search.run_bls(tt, ff, ee)
            vec, _ = features.extract(tt, ff, ee, res)
            X.append(vec); y.append(meta["label"])
        except Exception as e:
            skipped += 1
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(curves)}  (skipped {skipped})")
    return np.array(X), np.array(y)


def main(n_per_class=150):
    print(f"Generating synthetic training set ({n_per_class}/class)...")
    curves = synthetic.make_dataset(n_per_class=n_per_class, seed=1)
    print(f"  total curves: {len(curves)}")

    print("Extracting features (preprocess -> BLS -> features)...")
    X, y = build_feature_table(curves)
    print(f"  feature table: {X.shape}, classes: {sorted(set(y))}")

    print("Training classifier...")
    clf, scores = classify.train(X, y)

    report, cm, _ = classify.evaluate(clf, X, y)
    print("\n--- Training-set classification report ---")
    print(report)

    model_path = os.path.join(OUT, "classifier.joblib")
    classify.save(clf, model_path)
    np.savez(os.path.join(OUT, "training_features.npz"), X=X, y=y)
    cm_path = visualize.plot_confusion(cm, classify.CLASSES,
                                       os.path.join(OUT, "confusion_matrix.png"))
    print(f"\nSaved model -> {model_path}")
    print(f"Saved confusion matrix -> {cm_path}")
    return clf


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    main(n)

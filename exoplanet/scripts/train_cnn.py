#!/usr/bin/env python3
"""
train_cnn.py
------------
Train the AstroNet-style 1-D CNN on phase-folded views.

Pipeline per curve:  preprocess -> BLS -> make_views (global + local) -> CNN.

Saves the trained model to outputs/cnn_model.keras and a training-history plot.
Use --n to set curves per class (default 120).
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe import synthetic, preprocess, search, cnn

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def build_views(curves):
    G, L, Y = [], [], []
    skipped = 0
    for i, (t, flux, err, meta) in enumerate(curves):
        try:
            tt, ff, ee = preprocess.preprocess(t, flux, err)
            res = search.run_bls(tt, ff, ee)
            g, l = cnn.make_views(tt, ff, res)
            G.append(g); L.append(l); Y.append(meta["label"])
        except Exception:
            skipped += 1
        if (i + 1) % 100 == 0:
            print(f"  built views {i+1}/{len(curves)} (skipped {skipped})")
    return G, L, Y


def main(n_per_class=120, epochs=40):
    print(f"Generating training set ({n_per_class}/class)...")
    curves = synthetic.make_dataset(n_per_class=n_per_class, seed=7)
    print(f"  total curves: {len(curves)}")

    print("Building global+local views (preprocess -> BLS -> views)...")
    G, L, Y = build_views(curves)
    Xg, Xl, y = cnn.prepare_xy(G, L, Y)
    print(f"  global {Xg.shape}  local {Xl.shape}  labels {y.shape}")

    print("Building CNN...")
    model = cnn.build_model()
    model.summary()

    print("Training...")
    hist = cnn.train(model, Xg, Xl, y, epochs=epochs)

    path = os.path.join(OUT, "cnn_model.keras")
    cnn.save(model, path)
    print(f"Saved CNN -> {path}")

    # history plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist.history["loss"], label="train")
        ax[0].plot(hist.history["val_loss"], label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(hist.history["accuracy"], label="train")
        ax[1].plot(hist.history["val_accuracy"], label="val")
        ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "cnn_training.png"), dpi=110)
        print(f"Saved history -> {OUT}/cnn_training.png")
    except Exception as e:
        print("history plot skipped:", e)

    # final val accuracy
    val_acc = hist.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {val_acc*100:.1f}%")
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    main(args.n, args.epochs)

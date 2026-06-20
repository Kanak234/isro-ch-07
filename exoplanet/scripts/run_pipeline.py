#!/usr/bin/env python3
"""
run_pipeline.py
---------------
End-to-end analysis of one or more light curves:

  load -> preprocess/detrend -> BLS search -> (gate) -> features ->
  classify (with confidence) -> if transit: fit parameters with MCMC ->
  visualise -> append to results table.

USAGE
  # on a machine WITH internet, analyse real TESS targets:
  python run_pipeline.py --tic 307210830 231663901

  # offline demo on synthetic science data (mixed classes):
  python run_pipeline.py --demo 8
"""
import os, sys, argparse, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe import preprocess, search, features, classify, fit, visualize, synthetic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def _classify(clf, cnn_model, tt, ff, ee, res):
    """Tree-only, or tree+CNN ensemble if a CNN model is supplied."""
    vec, _ = features.extract(tt, ff, ee, res)
    if cnn_model is None:
        return classify.classify_one(clf, vec) + (vec,)
    from exopipe import cnn as cnnmod, ensemble
    g, l = cnnmod.make_views(tt, ff, res)
    label, conf, pdict = ensemble.classify_ensemble(
        vec, g, l, tree_clf=clf, cnn_model=cnn_model)
    return label, conf, pdict, vec


def analyse(t, flux, err, name, clf, do_fit=True, cnn_model=None):
    rec = {"name": name}
    tt, ff, ee = preprocess.preprocess(t, flux, err)

    res = search.run_bls(tt, ff, ee)
    rec["bls_period"] = res["period"]
    rec["bls_snr"] = res["snr_robust"]
    rec["bls_power"] = res["power"]

    if not search.has_significant_dip(res):
        rec["note"] = "no significant periodic dip"
        label, conf, pdict, vec = _classify(clf, cnn_model, tt, ff, ee, res)
        rec["label"], rec["confidence"], rec["probabilities"] = label, conf, pdict
    else:
        label, conf, pdict, vec = _classify(clf, cnn_model, tt, ff, ee, res)
        rec["label"], rec["confidence"], rec["probabilities"] = label, conf, pdict

    fit_params = None
    if do_fit and rec["label"] == "transit":
        print(f"    fitting transit parameters (MCMC)...")
        fit_params = fit.fit_transit(tt, ff, ee, res, nsteps=2000, burn=600)
        for k in ["period", "duration", "depth", "rp", "a", "inc"]:
            if k in fit_params and isinstance(fit_params[k], tuple):
                rec[f"fit_{k}"] = fit_params[k][0]
                rec[f"fit_{k}_err"] = fit_params[k][1]

    img = os.path.join(OUT, f"{name}.png")
    visualize.plot_result(t, flux, tt, ff, ee, res,
                          rec["label"], rec["confidence"], pdict,
                          fit_params=fit_params,
                          title=f"{name}", save_path=img)
    rec["plot"] = os.path.relpath(img, ROOT)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", nargs="*", help="TESS TIC IDs (needs internet)")
    ap.add_argument("--demo", type=int, default=0,
                    help="N synthetic science curves to analyse offline")
    ap.add_argument("--model", default=os.path.join(OUT, "classifier.joblib"))
    ap.add_argument("--cnn", nargs="?", const=os.path.join(OUT, "cnn_model.keras"),
                    default=None, help="use tree+CNN ensemble (optional path to .keras)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print("No trained model found. Run train_classifier.py first.")
        sys.exit(1)
    clf = classify.load(args.model)

    cnn_model = None
    if args.cnn:
        if os.path.exists(args.cnn):
            from exopipe import cnn as cnnmod
            cnn_model = cnnmod.load(args.cnn)
            print(f"Ensemble mode: tree + CNN ({args.cnn})")
        else:
            print(f"CNN model not found at {args.cnn}; using tree only.")

    results = []

    if args.demo > 0:
        print(f"=== DEMO: {args.demo} synthetic science curves ===")
        rng = np.random.default_rng(99)
        gens = list(synthetic.GENERATORS.items())
        for i in range(args.demo):
            label_true, gen = gens[i % len(gens)]
            t, flux, err, meta = gen(seed=int(rng.integers(1e9)))
            name = f"demo_{i:02d}_true-{label_true}"
            print(f"  [{i+1}/{args.demo}] {name}")
            rec = analyse(t, flux, err, name, clf, cnn_model=cnn_model)
            rec["true_label"] = label_true
            results.append(rec)
            print(f"      -> predicted: {rec['label']} ({rec['confidence']*100:.0f}%)")

    if args.tic:
        from exopipe import data
        for tic in args.tic:
            print(f"  downloading TIC {tic} ...")
            try:
                t, flux, err, meta = data.download_tess(tic)
                rec = analyse(t, flux, err, f"TIC{tic}", clf, cnn_model=cnn_model)
                results.append(rec)
                print(f"      -> {rec['label']} ({rec['confidence']*100:.0f}%)")
            except Exception as e:
                print(f"      ERROR: {type(e).__name__}: {e}")

    # save results table
    out_json = os.path.join(OUT, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results -> {out_json}")

    # accuracy on demo
    demo = [r for r in results if "true_label" in r]
    if demo:
        acc = np.mean([r["label"] == r["true_label"] for r in demo])
        print(f"Demo classification accuracy: {acc*100:.1f}%  ({len(demo)} curves)")


if __name__ == "__main__":
    main()

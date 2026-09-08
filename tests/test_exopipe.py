import pytest
import numpy as np
from exopipe import synthetic, preprocess, search, features, classify, ensemble, fit


def test_synthetic_transit():
    t, flux, err, meta = synthetic.make_transit(seed=42)
    assert len(t) > 100
    assert len(flux) == len(t)
    assert len(err) == len(t)
    assert meta["label"] == "transit"
    assert meta["period"] > 0
    assert meta["depth"] > 0


def test_synthetic_eclipse():
    t, flux, err, meta = synthetic.make_eclipse(seed=42)
    assert len(t) > 100
    assert len(flux) == len(t)
    assert meta["label"] == "eclipse"
    assert meta["depth"] > 0
    assert meta["secondary"] > 0


def test_synthetic_blend():
    t, flux, err, meta = synthetic.make_blend(seed=42)
    assert len(t) > 100
    assert len(flux) == len(t)
    assert meta["label"] == "blend"


def test_synthetic_other():
    t, flux, err, meta = synthetic.make_other(seed=42)
    assert len(t) > 100
    assert len(flux) == len(t)
    assert meta["label"] == "other"


def test_preprocess_sigma_clip_and_flatten():
    t = np.linspace(0, 10, 500)
    flux = 1.0 + 0.001 * np.sin(t)
    flux[50] = 5.0  # outlier
    err = np.full_like(t, 0.001)

    t_clean, f_clean, e_clean = preprocess.sigma_clip(t, flux, err, sigma=3)
    assert len(t_clean) < len(t), "Outlier should be clipped"

    flat, trend = preprocess.flatten(t_clean, f_clean)
    assert len(flat) == len(t_clean)
    assert np.isclose(np.median(flat), 1.0, atol=1e-3)


def test_bls_search():
    t = np.linspace(0, 20, 1000)
    per = 3.5
    dur = 0.2
    phase = (t % per) / per
    flux = np.ones_like(t)
    flux[phase < (dur / per)] -= 0.02
    err = np.full_like(t, 0.001)

    res = search.run_bls(t, flux, err, period_min=1.0, period_max=5.0, n_periods=200, n_durations=3)
    assert "period" in res
    assert "depth" in res
    assert "snr_robust" in res
    assert res["depth"] > 0


def test_features_extraction():
    t, flux, err, meta = synthetic.make_transit(seed=123)
    res = search.run_bls(t, flux, err, period_min=2.0, period_max=10.0, n_periods=100, n_durations=3)
    vec, feats = features.extract(t, flux, err, res)

    assert len(vec) == len(features.FEATURE_NAMES)
    assert isinstance(feats, dict)
    assert "bls_depth" in feats
    assert "primary_depth" in feats
    assert np.all(np.isfinite(vec))


def test_classify_one():
    feature_vec = np.array([0.01, 15.0, 20.0, 0.01, 0.0001, 0.01, 0.001, 0.1, 0.9, 0.0005, 20.0, 0.05])
    label, conf, pdict = classify.classify_one(None, feature_vec)

    assert label in classify.CLASSES
    assert 0.0 <= conf <= 1.0
    assert set(pdict.keys()) == set(classify.CLASSES)
    assert np.isclose(sum(pdict.values()), 1.0)


def test_ensemble_combination():
    p_tree = {"transit": 0.8, "eclipse": 0.1, "blend": 0.05, "other": 0.05}
    p_cnn = {"transit": 0.7, "eclipse": 0.2, "blend": 0.05, "other": 0.05}

    label, conf, pdict = ensemble.combine_proba(p_tree, p_cnn, w_tree=0.6, w_cnn=0.4)
    assert label == "transit"
    assert conf > 0.7
    assert np.isclose(sum(pdict.values()), 1.0)


def test_fit_transit():
    t = np.linspace(0, 10, 500)
    flux = np.ones_like(t)
    err = np.full_like(t, 0.001)
    res = {"period": 3.0, "t0": 1.0, "duration": 0.2, "depth": 0.01}

    params = fit.fit_transit(t, flux, err, res, run_mcmc=False)
    assert isinstance(params, dict)
    assert "depth" in params
    assert "duration" in params

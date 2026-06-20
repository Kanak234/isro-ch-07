# AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

A complete, end-to-end pipeline that detects periodic dips in noisy TESS light
curves, classifies them as **transit / eclipse / blend / other**, estimates
transit parameters (period, duration, depth) **with uncertainties**, and produces
diagnostic visualisations with a confidence level for every detection.

---

## 1. Install

```bash
pip install numpy scipy scikit-learn matplotlib
pip install lightkurve astropy batman-package emcee
```

(On managed systems add `--break-system-packages`.)

---

## 2. Project layout

```
exoplanet_pipeline/
├── exopipe/                 # the library
│   ├── synthetic.py         # physics-based light-curve generator (training + offline test)
│   ├── data.py              # REAL TESS download via lightkurve/MAST
│   ├── preprocess.py        # clean + transit-preserving detrend
│   ├── search.py            # Box Least Squares period search + SNR
│   ├── features.py          # 12 physical discriminating features
│   ├── classify.py          # gradient-boosted classifier (+confidence)
│   ├── cnn.py               # AstroNet-style 1-D CNN (global+local views)
│   ├── ensemble.py          # combines tree + CNN probabilities
│   ├── fit.py               # batman + emcee MCMC parameter fitting
│   └── visualize.py         # diagnostic plots
├── scripts/
│   ├── train_classifier.py  # train the gradient-boosted classifier
│   ├── train_cnn.py         # train the AstroNet-style CNN
│   ├── run_pipeline.py      # full analysis (tree-only or --cnn ensemble)
│   └── make_report.js       # builds the 3-page Word report
└── outputs/                 # models, plots, results.json, report
```

---

## 3. Quick start

### Step 1 — train the classifier
```bash
python scripts/train_classifier.py 80
```
Generates a labelled training set, extracts features, trains a
`HistGradientBoostingClassifier`, prints a 5-fold cross-validated macro-F1
score, and saves `outputs/classifier.joblib` + `outputs/confusion_matrix.png`.

> Achieved **CV macro-F1 = 0.97 ± 0.02** on synthetic data.

### Step 2a — run on REAL TESS data (needs internet)
```bash
python scripts/run_pipeline.py --tic 307210830 231663901
```
Downloads each TIC from MAST, runs the full pipeline, and writes a plot per
target plus `outputs/results.json`.

### Step 2b — offline demo on synthetic science data
```bash
python scripts/run_pipeline.py --demo 8
```
Analyses 8 mixed synthetic curves end-to-end and reports classification
accuracy (achieved **100%**) and recovered parameters.

### Step 3 (optional) — train the deep-learning model and use the ensemble
```bash
python scripts/train_cnn.py --n 120 --epochs 40      # train AstroNet-style CNN
python scripts/run_pipeline.py --demo 8 --cnn        # tree + CNN ensemble
```
The CNN reduces each curve to a **global view** (2001 bins, whole orbit) and a
**local view** (201 bins, zoomed on the transit), processes them through two
convolutional columns, and merges them for a 4-way classification. The
`--cnn` flag makes `run_pipeline` average the gradient-boosted and CNN
probabilities (the ensemble), which is how production vetting pipelines gain
robustness — the two models fail on different cases.

> CNN validation accuracy climbs steadily with training; the included demo
> model reached ~78% in 18 epochs on a small set and keeps rising with the
> recommended `--n 120 --epochs 40`.

> **Note on the CNN model file.** The trained gradient-boosted classifier
> (`classifier.joblib`) ships in `outputs/`. The CNN weights file
> (`cnn_model.keras`, ~110 MB) is **not** bundled to keep the download small —
> just run `python scripts/train_cnn.py` once to regenerate it (a few minutes
> on CPU). The pipeline runs tree-only without it; add `--cnn` once the file
> exists.

---

## 4. Using your own curated training catalogue

If you are given a labelled catalogue of confirmed planets, false positives and
eclipsing binaries, load each light curve as a tuple
`(t, flux, err, meta)` where `meta["label"]` is one of
`transit / eclipse / blend / other`, collect them into a list, and reuse the
same feature-building + training code in `scripts/train_classifier.py`
(replace `synthetic.make_dataset(...)` with your list). Everything downstream is
identical.

---

## 5. What each stage outputs

| Stage | Output |
|-------|--------|
| Search | period, t0, duration, depth, BLS power, depth-SNR, robust transit SNR |
| Classify | label + per-class probabilities (the confidence level) |
| Fit (transits) | period, duration, depth, Rp/Rs, a/Rs, inclination — each as median ±1σ from the MCMC posterior |
| Visualise | raw+detrended LC, BLS periodogram, phase-folded LC + model, text panel |

---

## 6. Scaling to a full sector (20–30k curves)

`exopipe.data.download_sector_targets(sector, limit=...)` yields light curves
one at a time so you can scan an entire sector without holding them all in
memory. For large runs, lower `n_periods` in `search.run_bls` during the first
detection pass, then re-fit survivors at high resolution.

---

## 7. Methods summary

- **Detrending:** iterative 5σ clip + Savitzky–Golay flattening with in-transit
  masking so real dips survive.
- **Search:** astropy Box Least Squares over a period/duration grid.
- **Classification:** gradient-boosted trees on physically-motivated features
  (secondary eclipse, odd–even depth difference, U-vs-V shape, depth-to-noise,
  duration/period). Optional AstroNet-style 1-D CNN on phase-folded views.
- **Parameters & uncertainties:** batman transit model fit by least squares,
  then emcee MCMC for full Bayesian posteriors (median ± 16th/84th percentile).

See `outputs/Exoplanet_Pipeline_Report.docx` for the full write-up.

---

## Team: Taaraka

Developed by **Team Taaraka** for the ISRO Bharatiya Antariksh Hackathon 2026 (BAH 2026), Challenge 07.

*Taaraka* (तारक) means "the guide" — the one that leads the way through the stars. Our pipeline guides astronomers through the noisy, crowded fields of TESS photometry to reliably uncover and characterise exoplanet transits.

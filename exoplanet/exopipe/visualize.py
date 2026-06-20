"""
visualize.py
------------
Diagnostic plots for each analysed light curve:
  - raw + detrended light curve
  - BLS periodogram with best period marked
  - phase-folded light curve with best-fit transit model overlaid
  - classification result + confidence + fitted parameters as a text panel
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_result(t_raw, flux_raw, t, flux, err, res, label, conf, pdict,
                fit_params=None, title="", save_path=None):
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])

    # (1) raw light curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_raw, flux_raw, ".", ms=1.5, alpha=0.4, color="gray", label="raw")
    ax1.plot(t, flux, ".", ms=1.5, alpha=0.6, color="C0", label="detrended")
    ax1.set_xlabel("Time [days]"); ax1.set_ylabel("Norm. flux")
    ax1.legend(loc="lower left", markerscale=4, fontsize=8)
    ax1.set_title(title or "Light curve", fontsize=11)

    # (2) BLS periodogram
    ax2 = fig.add_subplot(gs[1, 0])
    pg = res["periodogram"]
    ax2.plot(pg["period"], pg["power"], lw=0.7, color="C3")
    ax2.axvline(res["period"], color="k", ls="--", lw=1)
    ax2.set_xlabel("Period [days]"); ax2.set_ylabel("BLS power")
    ax2.set_title(f"BLS  P={res['period']:.4f} d", fontsize=10)

    # (3) phase-folded
    ax3 = fig.add_subplot(gs[1, 1])
    p, t0 = res["period"], res["t0"]
    phase = ((t - t0 + 0.5 * p) % p) / p - 0.5
    o = np.argsort(phase)
    ax3.plot(phase[o], flux[o], ".", ms=1.5, alpha=0.3, color="C0")
    # binned
    nb = 80
    edges = np.linspace(-0.5, 0.5, nb + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nb - 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    bmean = [np.median(flux[idx == b]) if (idx == b).any() else np.nan for b in range(nb)]
    ax3.plot(cen, bmean, "-", color="k", lw=1.2, label="binned")

    # model overlay
    if fit_params is not None and fit_params.get("method", "").startswith("batman"):
        try:
            from .fit import _model
            best = fit_params.get("_best")
            if best is not None:
                tt = np.linspace(t.min(), t.max(), 4000)
                mphase = ((tt - t0 + 0.5 * p) % p) / p - 0.5
                mo = np.argsort(mphase)
                mflux = _model(best, tt)
                ax3.plot(mphase[mo], mflux[mo], "-", color="C1", lw=1.5, label="fit")
        except Exception:
            pass
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_xlabel("Phase"); ax3.set_ylabel("Norm. flux")
    ax3.set_title("Phase-folded", fontsize=10)
    ax3.legend(fontsize=8, markerscale=3)

    # (4) text panel
    ax4 = fig.add_subplot(gs[2, :]); ax4.axis("off")
    lines = []
    lines.append(f"CLASSIFICATION:  {label.upper()}   (confidence {conf*100:.1f}%)")
    prob_str = "   ".join(f"{k}:{v*100:.0f}%" for k, v in
                          sorted(pdict.items(), key=lambda x: -x[1]))
    lines.append(f"class probabilities:  {prob_str}")
    lines.append(f"BLS depth-SNR: {res['snr_robust']:.1f}    BLS power: {res['power']:.1f}")
    lines.append("")
    if fit_params is not None:
        lines.append(f"FITTED PARAMETERS  (method: {fit_params.get('method','-')})")
        for key, unit in [("period", "d"), ("duration", "d"),
                          ("depth", ""), ("rp", "Rp/Rs"),
                          ("a", "a/Rs"), ("inc", "deg")]:
            if key in fit_params:
                v = fit_params[key]
                if isinstance(v, tuple):
                    mid, mlo, mhi = v
                    if np.isfinite(mlo):
                        if key == "depth":
                            lines.append(f"   {key:10s}= {mid*1e3:.3f} (+{mhi*1e3:.3f}/-{mlo*1e3:.3f}) ppt")
                        else:
                            lines.append(f"   {key:10s}= {mid:.5f} (+{mhi:.5f}/-{mlo:.5f}) {unit}")
                    else:
                        disp = mid*1e3 if key == "depth" else mid
                        u = "ppt" if key == "depth" else unit
                        lines.append(f"   {key:10s}= {disp:.5f} {u}")
    ax4.text(0.01, 0.98, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=9.5, transform=ax4.transAxes)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig


def plot_confusion(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Classifier confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return save_path

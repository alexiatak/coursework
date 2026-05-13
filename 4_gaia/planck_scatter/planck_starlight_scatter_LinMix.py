#!/usr/bin/env python3
"""
planck_starlight_scatter.py

Mehandiratta+2026 Fig. 2 -style scatter for the Markkanen field:
    Planck 353 GHz Q vs starlight q  (Galactic, IAU)
    Planck 353 GHz U vs starlight u  (Galactic, IAU)

Inputs:
    - merged_output.csv      RoboPol q,u in equatorial frame, IAU
    - planck_353_{I,Q,U}_smoothed20arcmin_nside1024_nested.fits

Outputs (in OUT_DIR):
    - markkanen_planck_starlight_scatter.png
    - planck_starlight_paired_robopol.csv     one row per RoboPol star
    - planck_starlight_pixavg_robopol.csv     one row per HEALPix pixel

Conventions:
    - Planck Q is read as-is.  Planck U is FLIPPED on read (COSMO -> IAU).
      After this, everything is IAU.
    - Optical q,u are rotated from equatorial to Galactic frame using the
      Appenzeller (1968) correction so they live in the same frame as Planck.
    - Submm dust EMISSION polarization is perpendicular to starlight
      ABSORPTION polarization, so we expect a NEGATIVE slope on both panels.
      |R| is reported as the absolute value of the slope.

Methodological note:
    Mehandiratta+2026 (https://www.aanda.org/articles/aa/pdf/2026/04/aa57681-25.pdf)
    fit Q-q and U-u using LinMix (Kelly 2007) for the per-panel slopes,
    with K = 2 Gaussian mixture prior on the true x distribution and broad
    non-informative priors on slope and intercept; plus a custom emcee MCMC
    for the joint R_P/p slope using the FULL Planck noise covariance
    (C_QQ, C_UU, off-diagonal C_QU) and the post-Galactic-rotation off-diagonal
    optical covariance sigma_qu.


    Two fit lines are plotted on each panel:
      (i)  LinMix with free intercept (matches paper exactly);
      (ii) iterative weighted total-least-squares forcing intercept = 0
           (sanity check that the relation passes through the origin).
 
Run:
    python planck_starlight_scatter.py
"""
from __future__ import annotations

# Requires the LinMix Bayesian linear-regression package (Kelly 2007 port).
# LinMix is NOT on PyPI; install from GitHub:
#     pip install git+https://github.com/jmeyers314/linmix.git
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp

try:
    import linmix
except ImportError:
    sys.exit(
        "ERROR: linmix not installed. Install with:\n"
        "    pip install git+https://github.com/jmeyers314/linmix.git"
    )


# CONFIG  -- edit these paths to match your setup

ROBOPOL_CSV = os.path.expanduser(
    "~/Desktop/coursework/2_sky_plot/merged_output.csv"
)

PANOPOULOU_CSV = os.path.expanduser(
    "~/Desktop/coursework/0_data/R/external_panopoulou_combined_polygon.csv"
)

PLANCK_DIR = os.path.expanduser("~/Desktop/coursework/3_planck")
PLANCK_I_FITS  = os.path.join(PLANCK_DIR, "planck_353_I_smoothed20arcmin_nside1024_nested.fits")
PLANCK_Q_FITS  = os.path.join(PLANCK_DIR, "planck_353_Q_smoothed20arcmin_nside1024_nested.fits")
PLANCK_U_FITS  = os.path.join(PLANCK_DIR, "planck_353_U_smoothed20arcmin_nside1024_nested.fits")
PLANCK_QQ_FITS = os.path.join(PLANCK_DIR, "planck_353_QQ_Cov_smoothed20arcmin_nside1024_nested.fits")
PLANCK_UU_FITS = os.path.join(PLANCK_DIR, "planck_353_UU_Cov_smoothed20arcmin_nside1024_nested.fits")

OUT_DIR = os.path.expanduser(
    "~/Desktop/coursework/4_gaia/planck_scatter/planck_output"
)

FIG_DPI = 220

# Fixed plot axis limits, applied identically to both panels (per Dima's
# request that the panels share a single scale).
X_LIMITS = (-0.023, 0.016)   # fractional q / u
Y_LIMITS = (-0.12, 0.15)     # MJy/sr

# LinMix mixture components for the prior on the true x distribution
# (Kelly 2007). Mehandiratta+2026 uses K = 2.
LINMIX_K = 2

# ---------------------------------------------------------------------------
# Unit conversion: K_CMB -> MJy/sr at 353 GHz (computed at run time).
# Set to None to compute automatically via astropy; or hardcode a float.
# The standard value is ~287.45 MJy/sr per K_CMB.
KCMB_TO_MJYSR = None   # computed in main() and passed to plot_scatter()

# Reference lines from the literature (plotted through the origin).
# Slopes are in MJy/sr per unit fractional polarization (not percent).
# Toggle either line off by setting the corresponding flag to False.
SHOW_PLANCK_REFLINE      = True   # Planck Coll. XII 2020: R_P/p = -5.42 MJy/sr
SHOW_MEHANDIRATTA_REFLINE = True  # Mehandiratta+2026: -5.13 (Q), -3.64 (U)
SHOW_PANOPOULOU           = True  # overlay Panopoulou+2025 stars as a second set

# Planck XII 2020 all-sky R_P/p (same value on both panels)
_PLANCK_SLOPE = -5.42          # MJy/sr

# Mehandiratta+2026 per-panel slopes (Q panel, U panel)
_MEHANDIRATTA_SLOPE_Q = -5.13  # MJy/sr
_MEHANDIRATTA_SLOPE_U = -3.64  # MJy/sr
# ---------------------------------------------------------------------------

# Optional rebinning of stars into coarser HEALPix pixels for averaging.
# The smoothed Planck map stays at native Nside (1024); we only change the
# pixelization used to group stars when computing per-pixel weighted means.
# Set to None to use the map's native Nside (current behavior, 1024).
# Try 512, 256, 128 .
BIN_NSIDE = None
#BIN_NSIDE = 512

# Warn loudly if the Planck FITS files are not in Galactic coordinates.
# The Appenzeller rotation in optical_to_galactic_qu() assumes Planck is
# in Galactic frame (COORDSYS = 'G').
CHECK_COORDSYS = True

# Output filename suffix when BIN_NSIDE differs from the map's native Nside.
# Lets us produce side-by-side variants without overwriting earlier outputs.
def _bin_tag(map_nside, bin_nside):
    if bin_nside is None or bin_nside == map_nside:
        return ""
    return f"_bin{bin_nside}"

# Coordinate / Stokes utilities

def appenzeller_rotation(ra_deg, dec_deg):
    """Return Dt (degrees) such that EVPA_galactic = EVPA_equatorial + Dt.

    Uses the Galactic-pole position (RA_GP=192.85948, Dec_GP=27.12825) in
    degrees, J2000.  Identical to the appenzeller() function used in
    final_plot.py.
    """
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    ra_gp = np.radians(192.85948)
    dec_gp = np.radians(27.12825)
    Dt = np.arctan2(
        np.sin(ra - ra_gp),
        np.cos(dec) * np.tan(dec_gp) - np.sin(dec) * np.cos(ra - ra_gp),
    )
    return np.degrees(Dt)


def optical_to_galactic_qu(q_pct, u_pct, sq_pct, su_pct, ra_deg, dec_deg):
    """Rotate optical q,u (equatorial IAU, percent) into the Galactic frame.

    Returns q_gal, u_gal, sq_gal, su_gal in percent.
    """
    q = np.asarray(q_pct, dtype=float)
    uu = np.asarray(u_pct, dtype=float)
    sq = np.asarray(sq_pct, dtype=float)
    su = np.asarray(su_pct, dtype=float)

    p = np.sqrt(q * q + uu * uu)
    evpa_eq = 0.5 * np.arctan2(uu, q)
    Dt_deg = appenzeller_rotation(ra_deg, dec_deg)
    evpa_gal = evpa_eq + np.radians(Dt_deg)

    cos2t = np.cos(2.0 * evpa_gal)
    sin2t = np.sin(2.0 * evpa_gal)
    q_gal = p * cos2t
    u_gal = p * sin2t

    p_safe = np.where(p > 0, p, np.nan)
    sp = np.sqrt((q * sq) ** 2 + (uu * su) ** 2) / p_safe
    sevpa = 0.5 * np.sqrt((uu * sq) ** 2 + (q * su) ** 2) / (p_safe ** 2)

    sq_gal = np.sqrt((cos2t * sp) ** 2 + (2.0 * p * sin2t * sevpa) ** 2)
    su_gal = np.sqrt((sin2t * sp) ** 2 + (2.0 * p * cos2t * sevpa) ** 2)
    return q_gal, u_gal, sq_gal, su_gal



# Planck sampling

def sample_planck(df_stars):
    """Sample smoothed Planck-353 I,Q,U at each star position.

    Returns a frame with columns:
        Name, ra, dec, l, b, pix,
        I_planck_KCMB, Q_planck_KCMB, U_planck_KCMB,
        sQ_planck_KCMB, sU_planck_KCMB

    sQ, sU are the per-pixel standard deviations from sqrt(QQ_Cov),
    sqrt(UU_Cov), both smoothed alongside the signal maps .

    NOTE: U_planck_KCMB is FLIPPED on read (COSMO -> IAU).  Q is unchanged.
    """
    for f in (PLANCK_I_FITS, PLANCK_Q_FITS, PLANCK_U_FITS,
              PLANCK_QQ_FITS, PLANCK_UU_FITS):
        if not os.path.isfile(f):
            sys.exit(f"ERROR: Planck FITS not found: {f}")
        if os.path.getsize(f) < 1000:
            sys.exit(f"ERROR: {f} looks like a git-lfs pointer file (<1 kB).")

    print(f"  reading I from {os.path.basename(PLANCK_I_FITS)}")
    I_map, I_hdr = hp.read_map(PLANCK_I_FITS, nest=True, h=True)
    print(f"  reading Q from {os.path.basename(PLANCK_Q_FITS)}")
    Q_map = hp.read_map(PLANCK_Q_FITS, nest=True)
    print(f"  reading U from {os.path.basename(PLANCK_U_FITS)} (flipping COSMO -> IAU)")
    U_map = -hp.read_map(PLANCK_U_FITS, nest=True)
    print(f"  reading QQ_Cov from {os.path.basename(PLANCK_QQ_FITS)}")
    QQ_Cov_map = hp.read_map(PLANCK_QQ_FITS, nest=True)
    print(f"  reading UU_Cov from {os.path.basename(PLANCK_UU_FITS)}")
    UU_Cov_map = hp.read_map(PLANCK_UU_FITS, nest=True)
    # Guard against tiny negative values from smoothing kernel ringing.
    QQ_Cov_map = np.clip(QQ_Cov_map, 0.0, None)
    UU_Cov_map = np.clip(UU_Cov_map, 0.0, None)
    nside = hp.get_nside(I_map)
    print(f"  Nside = {nside}")

    if CHECK_COORDSYS:
        coordsys = None
        for k, v in I_hdr:
            if str(k).strip().upper() == "COORDSYS":
                coordsys = str(v).strip().strip("'").strip()
                break
        if coordsys is None:
            print("  WARNING: COORDSYS keyword not found in Planck I FITS header.")
            print("           This script assumes Planck maps are in Galactic frame.")
        elif coordsys.upper() not in ("G", "GAL", "GALACTIC"):
            print(f"  WARNING: Planck COORDSYS = {coordsys!r}, not 'G'.")
            print("           Optical q,u are rotated to Galactic, so a non-G map")
            print("           will give wrong scatter slopes. Aborting.")
            sys.exit(1)
        else:
            print(f"  COORDSYS = {coordsys!r}  (Galactic frame, OK)")

    sc = SkyCoord(df_stars["ra"].to_numpy(float),
                  df_stars["dec"].to_numpy(float),
                  unit="deg", frame="icrs").galactic
    theta = np.radians(90.0 - sc.b.deg)
    phi = np.radians(sc.l.deg)
    pix = hp.ang2pix(nside, theta, phi, nest=True)

    out = pd.DataFrame({
        "Name": df_stars["Name"].values,
        "ra": df_stars["ra"].values,
        "dec": df_stars["dec"].values,
        "l": sc.l.deg,
        "b": sc.b.deg,
        "pix": pix,
        "I_planck_KCMB": I_map[pix],
        "Q_planck_KCMB": Q_map[pix],
        "U_planck_KCMB": U_map[pix],
        "sQ_planck_KCMB": np.sqrt(QQ_Cov_map[pix]),
        "sU_planck_KCMB": np.sqrt(UU_Cov_map[pix]),
    })
    return out, int(nside)



# Linear fits: LinMix (Kelly 2007) free-intercept, plus weighted TLS through origin

def linmix_fit(x, y, sx, sy, K=2, seed=42):
    """Bayesian linear fit y = alpha + beta*x using LinMix (Kelly 2007).

    Matches the per-panel prescription in Mehandiratta+2026, Fig. 2:
        - K = 2 Gaussian mixture prior on the true x distribution;
        - broad non-informative priors on slope (beta) and intercept (alpha);
        - measurement errors sx, sy on both axes treated as Gaussian.

    Returns (alpha, beta, sa, sb, chi2_red). Posterior point estimates use
    mean of the chain; uncertainties are posterior standard deviations.
    chi2_red is computed from posterior-mean values for diagnostic purposes
    (LinMix already absorbs unmodelled scatter into the intrinsic-variance
    parameter, so no chi2 rescaling of sa, sb is applied).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float); sy = np.asarray(sy, dtype=float)
    ok = (np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy)
          & (sx > 0) & (sy > 0))
    x, y, sx, sy = x[ok], y[ok], sx[ok], sy[ok]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    lm = linmix.LinMix(x, y, xsig=sx, ysig=sy, K=K, seed=seed)
    lm.run_mcmc(silent=True)
    alpha = float(np.mean(lm.chain['alpha']))
    beta  = float(np.mean(lm.chain['beta']))
    sa    = float(np.std(lm.chain['alpha']))
    sb    = float(np.std(lm.chain['beta']))

    res = y - alpha - beta * x
    sres2 = sy ** 2 + beta ** 2 * sx ** 2
    chi2_red = float(np.sum(res ** 2 / sres2) / max(n - 2, 1))
    return alpha, beta, sa, sb, chi2_red


def fit_no_intercept(x, y, sx, sy, n_iter=200):
    """Iterative weighted total-least-squares forcing y = beta * x (intercept = 0).

    Effective per-point variance sigma_eff^2 = sy^2 + beta^2 * sx^2.
    Weights w = 1 / sigma_eff^2; beta = sum(w*x*y) / sum(w*x*x); iterated
    to convergence. Uncertainty sb is the formal 1-sigma from the final
    weighted normal equations.

    Returns (beta, sb).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float); sy = np.asarray(sy, dtype=float)
    ok = (np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy)
          & (sx > 0) & (sy > 0))
    x, y, sx, sy = x[ok], y[ok], sx[ok], sy[ok]
    n = len(x)
    if n < 2:
        return np.nan, np.nan

    beta = float(np.sum(x * y) / np.sum(x * x))
    for _ in range(n_iter):
        sig_eff2 = sy ** 2 + beta ** 2 * sx ** 2
        w = 1.0 / sig_eff2
        beta_new = float(np.sum(w * x * y) / np.sum(w * x * x))
        if abs(beta_new - beta) < 1e-12:
            beta = beta_new
            break
        beta = beta_new
    sig_eff2 = sy ** 2 + beta ** 2 * sx ** 2
    w = 1.0 / sig_eff2
    sb = float(1.0 / np.sqrt(np.sum(w * x * x)))
    return beta, sb


def average_per_pixel(df_paired, value_cols, err_cols):
    """Inverse-variance average rows that share a pixel index."""
    rows = []
    for pix, grp in df_paired.groupby("pix"):
        row = {"pix": int(pix), "n_stars": len(grp),
               "ra_mean": grp["ra"].mean(), "dec_mean": grp["dec"].mean(),
               "l_mean": grp["l"].mean(), "b_mean": grp["b"].mean()}
        for v, e in zip(value_cols, err_cols):
            vals = grp[v].to_numpy(float); errs = grp[e].to_numpy(float)
            mask = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
            if mask.sum() == 0:
                row[v] = np.nan; row[e] = np.nan; continue
            w = 1.0 / errs[mask] ** 2
            row[v] = float((w * vals[mask]).sum() / w.sum())
            row[e] = float(np.sqrt(1.0 / w.sum()))
        for col in ("Q_planck_KCMB", "U_planck_KCMB", "I_planck_KCMB",
                    "sQ_planck_KCMB", "sU_planck_KCMB"):
            if col in grp.columns:
                row[col] = float(grp[col].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)



def load_robopol(path):
    df = pd.read_csv(path)
    needed = {"Name", "ra", "dec", "q", "u", "sq", "su"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {missing}")
    if np.nanmax(np.abs(df[["q", "u"]].to_numpy(float))) < 0.1:
        for c in ("q", "u", "sq", "su"):
            df[c] = df[c] * 100.0
        print(f"  loaded {len(df)} stars; promoted q,u from fraction to percent")
    else:
        print(f"  loaded {len(df)} stars; q,u already in percent")
    return df




def plot_scatter(paired, pix_avg, outfile, kcmb_to_mjysr, paired_pano=None):
    """Two-panel Mehandiratta-style scatter with LinMix fits.

    Planck Q/U are converted from K_CMB to MJy/sr for the plot and the fit.
    Optical q/u are converted from percent to fraction for the plot and fit,
    so the fit slope is directly in MJy/sr (comparable to R_P/p literature).
    CSV outputs are NOT changed (they stay in K_CMB and percent).

    Each panel shows two fit lines:
      (i)  LinMix with free intercept (Kelly 2007, K = LINMIX_K), matching
           Mehandiratta+2026 Fig. 2;
      (ii) iterative weighted total-least-squares forcing intercept = 0.
    Both panels share the fixed X_LIMITS and Y_LIMITS axis ranges.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Reference slopes in MJy/sr per unit fraction; one entry per panel.
    ref_slopes = [
        # (planck_ref_slope, mehandiratta_slope)
        (_PLANCK_SLOPE, _MEHANDIRATTA_SLOPE_Q),   # Q panel
        (_PLANCK_SLOPE, _MEHANDIRATTA_SLOPE_U),   # U panel
    ]

    panels = [
        (axes[0], "q_gal", "sq_gal",
         r"$q_{\rm v}^{\rm gal}$  [fraction]",
         "Q_planck_KCMB", "sQ_planck_KCMB",
         r"$Q_{\rm 353}$  [MJy sr$^{-1}$]",
         ref_slopes[0]),
        (axes[1], "u_gal", "su_gal",
         r"$u_{\rm v}^{\rm gal}$  [fraction]",
         "U_planck_KCMB", "sU_planck_KCMB",
         r"$U_{\rm 353}$  [MJy sr$^{-1}$]",
         ref_slopes[1]),
    ]

    fit_summary = {}
    for (ax, qcol, scol, qlabel, planckcol, planck_scol, planckname,
         (sl_planck, sl_meh)) in panels:

        # Convert units for plotting/fitting only (do not modify the DataFrames).
        x_data = pix_avg[qcol].to_numpy(float) / 100.0          # percent -> fraction
        x_err  = pix_avg[scol].to_numpy(float) / 100.0
        y_data = pix_avg[planckcol].to_numpy(float) * kcmb_to_mjysr  # K_CMB -> MJy/sr
        y_err  = pix_avg[planck_scol].to_numpy(float) * kcmb_to_mjysr  # same conversion

        # per-pixel points (now with real y-errors from sqrt(QQ_Cov)/sqrt(UU_Cov))
        ax.errorbar(x_data, y_data,
                    xerr=x_err, yerr=y_err, fmt="o", ms=6,
                    color="forestgreen", alpha=1.0, mec="black", mew=0.4,
                    label=f"per pixel (N={len(pix_avg)})", zorder=2)

        # Optional Panopoulou+2025 overlay (not used in the fit).
        if paired_pano is not None and len(paired_pano) > 0:
            xp = paired_pano[qcol].to_numpy(float) / 100.0
            xpe = paired_pano[scol].to_numpy(float) / 100.0
            yp = paired_pano[planckcol].to_numpy(float) * kcmb_to_mjysr
            ype = paired_pano[planck_scol].to_numpy(float) * kcmb_to_mjysr
            ax.errorbar(xp, yp, xerr=xpe, yerr=ype, fmt="s", ms=4,
                        color="darkorange", alpha=0.7, mec="black", mew=0.3,
                        label=f"Panopoulou+2025 (N={len(paired_pano)})",
                        zorder=4)

        a, b, sa, sb, chi2_red = linmix_fit(x_data, y_data, x_err, y_err,
                                            K=LINMIX_K)
        b0, sb0 = fit_no_intercept(x_data, y_data, x_err, y_err)

        # x array for plotting fit lines, spanning the full panel.
        x_ref = np.array(X_LIMITS, dtype=float)

        # (i) LinMix free-intercept fit (matches Mehandiratta+2026 exactly).
        if np.isfinite(b):
            ax.plot(x_ref, a + b * x_ref, color="black", lw=1.4, zorder=4,
                    label=f"LinMix (free b): slope = {b:.2f} \u00b1 {sb:.2f} MJy/sr")

        # (ii) Weighted TLS with intercept fixed to 0.
        if np.isfinite(b0):
            ax.plot(x_ref, b0 * x_ref, color="purple", lw=1.2, ls=":", zorder=4,
                    label=f"LinMix-style, b = 0: slope = {b0:.2f} \u00b1 {sb0:.2f} MJy/sr")

        # Reference lines through the origin, spanning the full panel.
        if SHOW_PLANCK_REFLINE:
            ax.plot(x_ref, sl_planck * x_ref,
                    color="steelblue", lw=1.2, ls="--", zorder=2,
                    label=fr"Planck XII 2020: $R_{{P/p}}$ = {sl_planck:.2f} MJy/sr")
        if SHOW_MEHANDIRATTA_REFLINE:
            ax.plot(x_ref, sl_meh * x_ref,
                    color="tomato", lw=1.2, ls="-.", zorder=2,
                    label=fr"Mehandiratta+2026: $R_{{P/p}}$ = {sl_meh:.2f} MJy/sr")

        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.axvline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlim(X_LIMITS)
        ax.set_ylim(Y_LIMITS)
        ax.set_xlabel(qlabel)
        ax.set_ylabel(planckname)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
        fit_summary[qcol] = {"a": a, "b": b, "sa": sa, "sb": sb,
                             "chi2_red": chi2_red,
                             "b0": b0, "sb0": sb0}

    axes[0].set_title("Planck 353 GHz Q vs starlight q (Galactic, IAU)")
    axes[1].set_title("Planck 353 GHz U vs starlight u (Galactic, IAU)")
    fig.suptitle(
        "Markkanen field: submm dust emission vs optical absorption polarization\n"
        "(emission $\\perp$ absorption in dust  $\\Rightarrow$  expect negative slope)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fit_summary



def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Planck-353 vs RoboPol starlight scatter (Mehandiratta+2026 Fig. 2)")
    print("=" * 70)

    print(f"\n[1/4] Loading RoboPol from {ROBOPOL_CSV}")
    df = load_robopol(ROBOPOL_CSV)

    print("\n[2/4] Sampling Planck at star positions")
    df_planck, map_nside =sample_planck(df)

    print("\n[3/4] Rotating optical q,u into Galactic frame and pairing")
    q_gal, u_gal, sq_gal, su_gal = optical_to_galactic_qu(
        df["q"].to_numpy(float),
        df["u"].to_numpy(float),
        df["sq"].to_numpy(float),
        df["su"].to_numpy(float),
        df["ra"].to_numpy(float),
        df["dec"].to_numpy(float),
    )
    paired = pd.DataFrame({
        "Name": df["Name"].values,
        "ra": df["ra"].values,
        "dec": df["dec"].values,
        "l": df_planck["l"].values,
        "b": df_planck["b"].values,
        "pix": df_planck["pix"].values,
        "q_gal": q_gal, "u_gal": u_gal,
        "sq_gal": sq_gal, "su_gal": su_gal,
        "I_planck_KCMB": df_planck["I_planck_KCMB"].values,
        "Q_planck_KCMB": df_planck["Q_planck_KCMB"].values,
        "U_planck_KCMB": df_planck["U_planck_KCMB"].values,
        "sQ_planck_KCMB": df_planck["sQ_planck_KCMB"].values,
        "sU_planck_KCMB": df_planck["sU_planck_KCMB"].values,
    })

    # Optional coarser binning.
    if BIN_NSIDE is not None and BIN_NSIDE != map_nside:
        if BIN_NSIDE > map_nside:
            print(f"  WARNING: BIN_NSIDE={BIN_NSIDE} > map Nside={map_nside}, "
                  f"keeping map Nside.")
            bin_pix = paired["pix"].values
            effective_bin_nside = map_nside
        else:
            print(f"  Rebinning stars into Nside={BIN_NSIDE} pixels "
                  f"(map stays at Nside={map_nside})")
            theta = np.radians(90.0 - paired["b"].to_numpy(float))
            phi = np.radians(paired["l"].to_numpy(float))
            bin_pix = hp.ang2pix(BIN_NSIDE, theta, phi, nest=True)
            effective_bin_nside = BIN_NSIDE
    else:
        bin_pix = paired["pix"].values
        effective_bin_nside = map_nside

    paired["pix_bin"] = bin_pix
    # average_per_pixel groups on column "pix", so pass a renamed copy
    paired_for_bin = paired.rename(columns={"pix": "pix_native",
                                            "pix_bin": "pix"})
    pix_avg = average_per_pixel(paired_for_bin,
                                value_cols=["q_gal", "u_gal"],
                                err_cols=["sq_gal", "su_gal"])
    # 1/sigma^2 weighting 
    # see w = 1.0 / errs[mask]**2).
    print(f"  {len(paired)} stars -> {len(pix_avg)} unique bins at Nside="
          f"{effective_bin_nside} "
          f"({(paired_for_bin['pix'].value_counts() > 1).sum()} bins with >1 star)")
    
    print(f"  {len(paired)} stars -> {len(pix_avg)} unique pixels "
          f"({(paired['pix'].value_counts() > 1).sum()} pixels with >1 star)")

    print("\n[4/4] Plotting and writing outputs")

    # Compute K_CMB -> MJy/sr at 353 GHz using astropy thermodynamic equivalency.
    global KCMB_TO_MJYSR
    if KCMB_TO_MJYSR is None:
        from astropy.cosmology import Planck15
        freq_353 = 353e9 * u.Hz
        equiv = u.thermodynamic_temperature(freq_353, Planck15.Tcmb0)
        KCMB_TO_MJYSR = float((1.0 * u.K).to(u.MJy / u.sr, equivalencies=equiv).value)
    print(f"  K_CMB -> MJy/sr at 353 GHz: {KCMB_TO_MJYSR:.4f} MJy/sr per K_CMB")

    # Optionally build the Panopoulou+2025 overlay (same pipeline as RoboPol,
    # but no pixel-averaging; raw per-star points are plotted as the overlay).
    paired_pano = None
    if SHOW_PANOPOULOU and os.path.isfile(PANOPOULOU_CSV):
        print(f"\n  Loading Panopoulou+2025 from {PANOPOULOU_CSV}")        
        df_pano_raw = pd.read_csv(PANOPOULOU_CSV)
        df_pano_raw = df_pano_raw.dropna(subset=["p", "evpa", "e_p", "e_evpa"])
            # p is a fraction (0-1); convert to percent to match RoboPol units.
        p_pct = df_pano_raw["p"].to_numpy(float) * 100.0
        ep_pct = df_pano_raw["e_p"].to_numpy(float) * 100.0
        psi_rad = np.deg2rad(df_pano_raw["evpa"].to_numpy(float))
        epsi_rad = np.deg2rad(df_pano_raw["e_evpa"].to_numpy(float))
        q_eq = p_pct * np.cos(2.0 * psi_rad)
        u_eq = p_pct * np.sin(2.0 * psi_rad)
            # error propagation for q = p*cos(2psi), u = p*sin(2psi)
        sq_eq = np.sqrt((np.cos(2*psi_rad)*ep_pct)**2 + (2*p_pct*np.sin(2*psi_rad)*epsi_rad)**2)
        su_eq = np.sqrt((np.sin(2*psi_rad)*ep_pct)**2 + (2*p_pct*np.cos(2*psi_rad)*epsi_rad)**2)
        df_pano = pd.DataFrame({
            "Name": df_pano_raw["starID"].astype(str).values,
            "ra": df_pano_raw["RA"].to_numpy(float),
            "dec": df_pano_raw["DEC"].to_numpy(float),
            "q": q_eq, "u": u_eq,
            "sq": sq_eq, "su": su_eq,
            })
        df_pano_planck, _ = sample_planck(df_pano)
        
        q_gal_p, u_gal_p, sq_gal_p, su_gal_p = optical_to_galactic_qu(
            df_pano["q"].to_numpy(float),
            df_pano["u"].to_numpy(float),
            df_pano["sq"].to_numpy(float),
            df_pano["su"].to_numpy(float),
            df_pano["ra"].to_numpy(float),
            df_pano["dec"].to_numpy(float),
        )
        paired_pano = pd.DataFrame({
            "Name": df_pano["Name"].values,
            "ra": df_pano["ra"].values,
            "dec": df_pano["dec"].values,
            "l": df_pano_planck["l"].values,
            "b": df_pano_planck["b"].values,
            "pix": df_pano_planck["pix"].values,
            "q_gal": q_gal_p, "u_gal": u_gal_p,
            "sq_gal": sq_gal_p, "su_gal": su_gal_p,
            "I_planck_KCMB": df_pano_planck["I_planck_KCMB"].values,
            "Q_planck_KCMB": df_pano_planck["Q_planck_KCMB"].values,
            "U_planck_KCMB": df_pano_planck["U_planck_KCMB"].values,
            "sQ_planck_KCMB": df_pano_planck["sQ_planck_KCMB"].values,
            "sU_planck_KCMB": df_pano_planck["sU_planck_KCMB"].values,
        })
        print(f"  Panopoulou overlay: {len(paired_pano)} stars")
    elif SHOW_PANOPOULOU:
        print(f"\n  SHOW_PANOPOULOU is True but {PANOPOULOU_CSV} not found; skipping.")

    tag = _bin_tag(map_nside, effective_bin_nside)
    out_png = os.path.join(OUT_DIR,
                           f"markkanen_planck_starlight_scatter{tag}.png")
    fits = plot_scatter(paired, pix_avg, out_png, KCMB_TO_MJYSR,
                        paired_pano=paired_pano)
    paired.to_csv(os.path.join(OUT_DIR,
                               f"planck_starlight_paired_robopol{tag}.csv"),
                  index=False)
    pix_avg.to_csv(os.path.join(OUT_DIR,
                                f"planck_starlight_pixavg_robopol{tag}.csv"),
                   index=False)
    if paired_pano is not None:
        paired_pano.to_csv(os.path.join(OUT_DIR,
                                       f"planck_starlight_paired_panopoulou{tag}.csv"),
                          index=False)

    print(f"\n  wrote {out_png}")
    print(f"  wrote planck_starlight_paired_robopol{tag}.csv "
          f"({len(paired)} rows)")
    print(f"  wrote planck_starlight_pixavg_robopol{tag}.csv "
          f"({len(pix_avg)} rows)")

    print("\n" + "=" * 70)
    print("Fit summary  (slopes from per-pixel data, units: MJy/sr per fraction)")
    print("=" * 70)
    for label, key, sign in [("Q vs q", "q_gal", "Q"), ("U vs u", "u_gal", "U")]:
        f = fits[key]
        print(f" {label}:")
        print(f"   LinMix (free b):  slope = {f['b']:+.3e} \u00b1 {f['sb']:.1e} "
              f"({abs(f['b']) / f['sb']:.1f} sigma)")
        print(f"                     intercept = {f['a']:+.3e} \u00b1 {f['sa']:.1e} "
              f"({abs(f['a']) / f['sa']:.1f} sigma from zero)")
        print(f"   b = 0 (TLS):      slope = {f['b0']:+.3e} \u00b1 {f['sb0']:.1e} "
              f"({abs(f['b0']) / f['sb0']:.1f} sigma)")
        print(f"   chi2_red (free b) = {f['chi2_red']:.2f}")


    print("\nDone.")


if __name__ == "__main__":
    main()

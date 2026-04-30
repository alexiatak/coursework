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

  
Methodological note :
    Mehandiratta 2026 (https://www.aanda.org/articles/aa/pdf/2026/04/aa57681-25.pdf) fit Q-q and U-u using LinMix (Kelly 2007)
    for the per-panel slopes and a custom emcee MCMC for the joint R_P/p slope,
    with the FULL Planck noise covariance (C_QQ, C_UU, off-diagonal C_QU) and
    the post-Galactic-rotation off-diagonal optical covariance sigma_qu.
 
    This script does NOT do that.  It does a per-panel York (1969) fit because:
      a. we have RoboPol per-star sigma_q, sigma_u but NOT the Planck per-pixel
          noise covariance maps on disk - only the smoothed I, Q, U intensity
          maps - so a method that needs C_QQ, C_UU, C_QU is not runnable here;
      b. we drop the off-diagonal sigma_qu introduced by the equatorial->
          Galactic rotation in optical_to_galactic_qu();
      c. we do NOT do a joint fit across both panels.
 
Run:
    python planck_starlight_scatter.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp


# CONFIG  -- edit these paths to match your setup

ROBOPOL_CSV = os.path.expanduser(
    "~/Desktop/coursework/2_sky_plot/merged_output.csv"
)

PLANCK_DIR = os.path.expanduser("~/Desktop/coursework/3_planck")
PLANCK_I_FITS = os.path.join(PLANCK_DIR, "planck_353_I_smoothed20arcmin_nside1024_nested.fits")
PLANCK_Q_FITS = os.path.join(PLANCK_DIR, "planck_353_Q_smoothed20arcmin_nside1024_nested.fits")
PLANCK_U_FITS = os.path.join(PLANCK_DIR, "planck_353_U_smoothed20arcmin_nside1024_nested.fits")

OUT_DIR = os.path.expanduser(
    "~/Desktop/coursework/4_gaia/planck_scatter/planck_output"
)

FIG_DPI = 220



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
        I_planck_KCMB, Q_planck_KCMB, U_planck_KCMB

    NOTE: U_planck_KCMB is FLIPPED on read (COSMO -> IAU).  Q is unchanged.
    """
    for f in (PLANCK_I_FITS, PLANCK_Q_FITS, PLANCK_U_FITS):
        if not os.path.isfile(f):
            sys.exit(f"ERROR: Planck FITS not found: {f}")
        if os.path.getsize(f) < 1000:
            sys.exit(f"ERROR: {f} looks like a git-lfs pointer file (<1 kB).")

    print(f"  reading I from {os.path.basename(PLANCK_I_FITS)}")
    I_map = hp.read_map(PLANCK_I_FITS, nest=True)
    print(f"  reading Q from {os.path.basename(PLANCK_Q_FITS)}")
    Q_map = hp.read_map(PLANCK_Q_FITS, nest=True)
    print(f"  reading U from {os.path.basename(PLANCK_U_FITS)} (flipping COSMO -> IAU)")
    U_map = -hp.read_map(PLANCK_U_FITS, nest=True)
    nside = hp.get_nside(I_map)
    print(f"  Nside = {nside}")

    sc = SkyCoord(df_stars["ra"].to_numpy(float),
                  df_stars["dec"].to_numpy(float),
                  unit="deg", frame="icrs").galactic
    theta = np.radians(90.0 - sc.b.deg)
    phi = np.radians(sc.l.deg)
    pix = hp.ang2pix(nside, theta, phi, nest=True)

    return pd.DataFrame({
        "Name": df_stars["Name"].values,
        "ra": df_stars["ra"].values,
        "dec": df_stars["dec"].values,
        "l": sc.l.deg,
        "b": sc.b.deg,
        "pix": pix,
        "I_planck_KCMB": I_map[pix],
        "Q_planck_KCMB": Q_map[pix],
        "U_planck_KCMB": U_map[pix],
    })



# Linear fit (York 1969) and pixel averaging

def york_fit(x, y, sx, sy, n_iter=50):
  
    """Weighted linear fit y = a + b*x with errors on both axes.
    Returns (a, b, sa, sb_formal, chi2_red).  No correlation between sx, sy.
 
    Why York here (and not the LinMix/emcee approach used in
    Mehandiratta+2026): we do not have the Planck per-pixel noise
    covariance maps loaded, only the smoothed intensity maps, so a method
    that needs C_QQ, C_UU, C_QU is not runnable.  York with real sigma_x
    (RoboPol) and a small dummy sigma_y is the right cheap proxy: it
    reduces to weighted-x fitting in this regime, correctly down-weights
    points with large optical errors (Mark_65 concern), and gives an
    unbiased slope.  The reported slope error is rescaled to
    max(sb_formal, sb_resid) by the caller in plot_scatter() to be honest
    about residual scatter. 
    """
    
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    sx = np.asarray(sx, dtype=float); sy = np.asarray(sy, dtype=float)
    ok = (np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy)
          & (sx > 0) & (sy > 0))
    x, y, sx, sy = x[ok], y[ok], sx[ok], sy[ok]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    wx = 1.0 / sx ** 2
    wy = 1.0 / sy ** 2
    b = np.polyfit(x, y, 1)[0]
    for _ in range(n_iter):
        W = wx * wy / (wx + b * b * wy)
        Wsum = W.sum()
        xbar = (W * x).sum() / Wsum
        ybar = (W * y).sum() / Wsum
        U = x - xbar; V = y - ybar
        beta = W * (U / wy + b * V / wx)
        denom = (W * beta * U).sum()
        if denom == 0:
            break
        b_new = (W * beta * V).sum() / denom
        if abs(b_new - b) < 1e-12:
            b = b_new
            break
        b = b_new
    a = ybar - b * xbar
    W = wx * wy / (wx + b * b * wy)
    Wsum = W.sum()
    xbar = (W * x).sum() / Wsum
    U = x - xbar
    sb2 = 1.0 / (W * U * U).sum()
    sa2 = 1.0 / Wsum + xbar * xbar * sb2
    res = y - a - b * x
    sres2 = sy ** 2 + b * b * sx ** 2
    chi2_red = float(np.sum(res ** 2 / sres2) / max(n - 2, 1))
    return float(a), float(b), float(np.sqrt(sa2)), float(np.sqrt(sb2)), chi2_red


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
        for col in ("Q_planck_KCMB", "U_planck_KCMB", "I_planck_KCMB"):
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




def plot_scatter(paired, pix_avg, outfile):
    """Two-panel Mehandiratta-style scatter with York fit."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    panels = [
        (axes[0], "q_gal", "sq_gal", r"$q_{\rm v}^{\rm gal}$  [%]",
         "Q_planck_KCMB", r"$Q_{\rm 353}$  [K$_{\rm CMB}$]"),
        (axes[1], "u_gal", "su_gal", r"$u_{\rm v}^{\rm gal}$  [%]",
         "U_planck_KCMB", r"$U_{\rm 353}$  [K$_{\rm CMB}$]"),
    ]

    fit_summary = {}
    for ax, qcol, scol, qlabel, planckcol, planckname in panels:
        
        # per-pixel (strong)
        ax.errorbar(pix_avg[qcol], pix_avg[planckcol],
                    xerr=pix_avg[scol], fmt="o", ms=6,
                    color="forestgreen", alpha=1.0, mec="black", mew=0.4,
                    label=f"per pixel (N={len(pix_avg)})", zorder=3)

        x_data = pix_avg[qcol].to_numpy(float)
        y_data = pix_avg[planckcol].to_numpy(float)
        x_err = pix_avg[scol].to_numpy(float)
        y_err_dummy = np.full_like(y_data,
                                   max(0.01 * np.nanstd(y_data), 1e-12))
        a, b, sa, sb_formal, chi2_red = york_fit(x_data, y_data, x_err,
                                                 y_err_dummy)

        sb = sb_formal
        if np.isfinite(b):
            ok = np.isfinite(x_data) & np.isfinite(y_data)
            xx_ok = x_data[ok]; yy_ok = y_data[ok]
            res = yy_ok - (a + b * xx_ok)
            n_ok = len(xx_ok)
            if n_ok > 2:
                Sxx = np.sum((xx_ok - xx_ok.mean()) ** 2)
                if Sxx > 0:
                    s_res = np.sqrt(np.sum(res ** 2) / (n_ok - 2))
                    sb_resid = s_res / np.sqrt(Sxx)
                    sb = max(sb_formal, sb_resid)
            xx = np.linspace(np.nanmin(x_data) - 0.05,
                             np.nanmax(x_data) + 0.05, 50)
            ax.plot(xx, a + b * xx, color="black", lw=1.2, zorder=4,
                    label=f"slope = {b:.2e} \u00b1 {sb:.1e}\n|R| = {abs(b):.2e}")

        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.axvline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel(qlabel)
        ax.set_ylabel(planckname)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
        fit_summary[qcol] = {"a": a, "b": b, "sa": sa, "sb": sb,
                             "chi2_red": chi2_red}

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
    df_planck = sample_planck(df)

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
    })
    pix_avg = average_per_pixel(paired,
                                value_cols=["q_gal", "u_gal"],
                                err_cols=["sq_gal", "su_gal"])
    print(f"  {len(paired)} stars -> {len(pix_avg)} unique pixels "
          f"({(paired['pix'].value_counts() > 1).sum()} pixels with >1 star)")

    print("\n[4/4] Plotting and writing outputs")
    out_png = os.path.join(OUT_DIR, "markkanen_planck_starlight_scatter.png")
    fits = plot_scatter(paired, pix_avg, out_png)
    paired.to_csv(os.path.join(OUT_DIR, "planck_starlight_paired_robopol.csv"),
                  index=False)
    pix_avg.to_csv(os.path.join(OUT_DIR, "planck_starlight_pixavg_robopol.csv"),
                   index=False)

    print(f"\n  wrote {out_png}")
    print(f"  wrote planck_starlight_paired_robopol.csv "
          f"({len(paired)} rows)")
    print(f"  wrote planck_starlight_pixavg_robopol.csv "
          f"({len(pix_avg)} rows)")

    print("\n" + "=" * 70)
    print("Fit summary  (slopes from per-pixel data)")
    print("=" * 70)
    for label, key, sign in [("Q vs q", "q_gal", "Q"), ("U vs u", "u_gal", "U")]:
        f = fits[key]
        print(f"  {label}:  slope = {f['b']:+.3e} \u00b1 {f['sb']:.1e}   "
              f"|R| = {abs(f['b']):.3e}   "
              f"sigma = {abs(f['b']) / f['sb']:.1f} "
              f"(reduced chi2 = {f['chi2_red']:.2f})")

    print("\nDone.")


if __name__ == "__main__":
    main()

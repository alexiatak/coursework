#!/usr/bin/env python3
"""
pd_vs_reddening.py

Polarization fraction vs. reddening for the RoboPol Markkanen sample and the
Panopoulou+2025 polygon catalog, using the Planck GNILC dust opacity at 353 GHz
as the dust-column proxy.

Inputs:
    - merged_output.csv                 RoboPol q, u (fractional)
    - external_panopoulou_combined_polygon.csv   Panopoulou+2025 catalog
      or external_panopoulou_expanded_polygon.csv 
    - GNILC opacity map (Planck Legacy Archive product
      COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits), tau_353 in HDU 0,
      converted to E(B-V) via E(B-V) = 1.49e4 * tau_353
      (Planck Collaboration 2018, XII, Eq. 12)

Outputs (in OUT_DIR):
    - pd_vs_ebv_gnilc.png        3-panel figure
    - pd_vs_ebv_gnilc_table.csv  per-star table used in the plot
    - pd_vs_ebv_running_means.csv  running medians per sample
    - gnilc_pixel_histogram.png   Diagnose how the two samples share GNILC pixels

Run:
    python pd_vs_reddening.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import healpy as hp
from astropy.io import fits


# CONFIG. Edit paths and toggles here.
# Input tables.
ROBOPOL_CSV = os.path.expanduser(
    "~/Desktop/coursework/2_sky_plot/merged_output.csv"
)
PANOPOULOU_CSV = os.path.expanduser(
    "~/Desktop/coursework/0_data/R/external_panopoulou_expanded_polygon.csv"
)

# Reddening map.
# Path to the GNILC dust opacity full-sky FITS file. Kept outside the repo
# alongside the raw Planck 353 file.
GNILC_FITS = os.path.expanduser(
    "~/Desktop/extra_data/COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits"
)
# tau_353 -> E(B-V) conversion, Planck 2018 XII Eq. 12.
TAU_TO_EBV = 1.49e4
# Ordering used in the GNILC opacity file. Planck distributes these in NESTED.
GNILC_NEST = True

# Output directory.
OUT_DIR = os.path.expanduser(
    "~/Desktop/coursework/6_reddening_maps/output_pd_vs_reddening"
)

# Panopoulou filter selection. None or [] = keep all; otherwise list of FilterID
# values to keep (FilterID=0 is the main sample; 20 = Cousins R).
PANOPOULOU_FILTER_IDS: list[int] | None = None

# RoboPol outlier exclusion, kept consistent with markkanen_fig8_cloud_test.py.
EXCLUDE_STARS: list[str] = []
# EXCLUDE_STARS = ["Mark_65"]

# Signal-to-noise cut on p. None disables it. 
#A value of 2 or 3 cleans up the panopoulou low-p tail considerably.
MIN_PSNR: float | None = None

# Running mean: number of stars per equal-count bin. Reduce if a sample is too
# small to fill more than ~3 bins.
RUNNING_BIN_COUNT = 15

# Panopoulou distance split (Bailer-Jones r_med_photogeo, in pc).
# Stars with d < FOREGROUND_DIST_MAX are classified "foreground" (in front of
# the cloud), stars with d > BACKGROUND_DIST_MIN are classified "background"
# (behind the cloud, like RoboPol targets). Stars in between are dropped from
# the split panel but kept in the all-Panopoulou running mean.
FOREGROUND_DIST_MAX_PC = 150.0
BACKGROUND_DIST_MIN_PC = 200.0
# Smaller bin count for the distance-split subsamples since each is ~half the
# size of the full Panopoulou sample.
RUNNING_BIN_COUNT_SPLIT = 10

# Output filename suffix toggled automatically by the config.
def _suffix() -> str:
    parts = []
    if PANOPOULOU_FILTER_IDS:
        parts.append("F" + "-".join(str(f) for f in PANOPOULOU_FILTER_IDS))
    if EXCLUDE_STARS:
        parts.append("excl_" + "_".join(s.replace("Mark_", "M") for s in EXCLUDE_STARS))
    if MIN_PSNR is not None:
        parts.append(f"snr{MIN_PSNR:g}")
    return ("_" + "_".join(parts)) if parts else ""

SUFFIX = _suffix()
OUT_FIG = os.path.join(OUT_DIR, f"pd_vs_ebv_gnilc{SUFFIX}.png")
OUT_TABLE = os.path.join(OUT_DIR, f"pd_vs_ebv_gnilc_table{SUFFIX}.csv")
OUT_RUNNING = os.path.join(OUT_DIR, f"pd_vs_ebv_running_means{SUFFIX}.csv")
OUT_PIXEL_HIST = os.path.join(OUT_DIR, f"gnilc_pixel_histogram{SUFFIX}.png")

FIG_DPI = 220


def read_table_auto(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_robopol(path: str) -> pd.DataFrame:
    """Load RoboPol q, u, sq, su, ra, dec; promote q/u to percent if needed.

    Computes the modified asymptotic (MAS) estimator of p following
    Plaszczynski et al. 2014 to debias the Rician bias at low SNR. The MAS
    estimator collapses to the naive p at high SNR and is well-behaved at low
    SNR, so it is a safe default for both bright and faint stars.
    """
    df = read_table_auto(path)
    needed = ["Name", "q", "sq", "u", "su", "ra", "dec"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} missing columns: {missing}")

    for c in ["q", "sq", "u", "su", "ra", "dec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["q", "sq", "u", "su", "ra", "dec"]).copy()

    # Promote fractional q,u to percent, matching the rest of the repo.
    if max(df["q"].abs().max(), df["u"].abs().max()) < 0.1:
        for c in ["q", "sq", "u", "su"]:
            df[c] = df[c] * 100.0
        print(f"  RoboPol: {len(df)} stars, q/u promoted to percent")
    else:
        print(f"  RoboPol: {len(df)} stars, q/u already in percent")

    if EXCLUDE_STARS:
        before = len(df)
        df = df[~df["Name"].isin(EXCLUDE_STARS)].copy()
        print(f"  Excluded {before - len(df)} star(s): {EXCLUDE_STARS}")

    # Naive p and sigma_p.
    q = df["q"].to_numpy(float)
    u = df["u"].to_numpy(float)
    sq = df["sq"].to_numpy(float)
    su = df["su"].to_numpy(float)
    p_obs = np.sqrt(q * q + u * u)
    # Approximate sigma_p (valid when sq ~ su; good enough for plotting/SNR cut).
    sigma_p = np.sqrt((q * q * sq * sq + u * u * su * su) / np.where(p_obs > 0, p_obs * p_obs, 1.0))
    # MAS debiased p, Plaszczynski et al. 2014.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_mas = p_obs - sigma_p * sigma_p * (1.0 - np.exp(-(p_obs / sigma_p) ** 2)) / (2.0 * p_obs)
    p_mas = np.where(np.isfinite(p_mas) & (p_mas > 0), p_mas, 0.0)

    df["p_percent"] = p_mas
    df["sigma_p_percent"] = sigma_p
    df["sample"] = "RoboPol"
    df["FilterID"] = -1  # placeholder, RoboPol is R-band
    # RoboPol distances live in a separate BJ table; we do not split RoboPol by
    # distance here. These columns are present for concat compatibility only.
    df["r_med_photogeo"] = np.nan
    df["distance_class"] = "robopol"
    return df[["Name", "ra", "dec", "p_percent", "sigma_p_percent",
               "FilterID", "sample", "r_med_photogeo", "distance_class"]]


def load_panopoulou(path: str) -> pd.DataFrame:
    """Load Panopoulou+2025 polygon catalog and convert p from fractional to percent.

    Panopoulou+2025 already reports a debiased p in their pipeline, so we do
    not apply MAS again. Rows with missing p are dropped.
    """
    df = read_table_auto(path)
    # Normalize Dec column name (Panopoulou's exported CSVs sometimes use DEC).
    if "DEC" in df.columns and "Dec" not in df.columns:
        df = df.rename(columns={"DEC": "Dec"})
    needed = ["starID", "RA", "Dec", "p", "e_p", "FilterID"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} missing columns: {missing}")

    for c in ["RA", "Dec", "p", "e_p", "FilterID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Distance column for foreground/background split. Missing distances are
    # tolerated, those stars just get distance_class = "unknown".
    if "r_med_photogeo" in df.columns:
        df["r_med_photogeo"] = pd.to_numeric(df["r_med_photogeo"], errors="coerce")
    else:
        df["r_med_photogeo"] = np.nan
    df = df.dropna(subset=["RA", "Dec", "p", "e_p"]).copy()

    if PANOPOULOU_FILTER_IDS:
        before = len(df)
        df = df[df["FilterID"].isin(PANOPOULOU_FILTER_IDS)].copy()
        print(f"  Panopoulou: filter cut FilterID in {PANOPOULOU_FILTER_IDS}: "
              f"kept {len(df)}/{before}")

    # Panopoulou p is fractional, convert to percent.
    df["p_percent"] = df["p"].to_numpy(float) * 100.0
    df["sigma_p_percent"] = df["e_p"].to_numpy(float) * 100.0
    df["sample"] = "Panopoulou+2025"
    df = df.rename(columns={"starID": "Name", "Dec": "dec", "RA": "ra"})
    df["Name"] = df["Name"].astype(str)

    # Distance class for the split panel.
    d = df["r_med_photogeo"].to_numpy(float)
    cls = np.full(len(df), "unknown", dtype=object)
    cls[np.where(d < FOREGROUND_DIST_MAX_PC)[0]] = "foreground"
    cls[np.where(d > BACKGROUND_DIST_MIN_PC)[0]] = "background"
    cls[np.where((d >= FOREGROUND_DIST_MAX_PC) & (d <= BACKGROUND_DIST_MIN_PC))[0]] = "middle"
    df["distance_class"] = cls

    n_fg = int((cls == "foreground").sum())
    n_bg = int((cls == "background").sum())
    n_mid = int((cls == "middle").sum())
    n_unk = int((cls == "unknown").sum())
    print(f"  Panopoulou: {len(df)} stars after cleaning")
    print(f"    distance split: foreground (d<{FOREGROUND_DIST_MAX_PC:.0f} pc)={n_fg}, "
          f"middle={n_mid}, background (d>{BACKGROUND_DIST_MIN_PC:.0f} pc)={n_bg}, unknown={n_unk}")
    return df[["Name", "ra", "dec", "p_percent", "sigma_p_percent",
               "FilterID", "sample", "r_med_photogeo", "distance_class"]]


def load_gnilc_ebv(path: str) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Load the GNILC dust opacity map and return (E(B-V), sigma_E(B-V), nside, nest)."""
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: GNILC opacity FITS not found at {path}. "
            "Download COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits "
            "from the Planck Legacy Archive (https://pla.esac.esa.int/pla/#maps)."
        )
    # hp.read_map handles both NEST and RING automatically when given nest=None;
    # we keep tau in NEST since that is how Planck distributes it.

    try:
        tau = hp.read_map(path, field=0, nest=GNILC_NEST)
    except TypeError:
        tau = hp.read_map(path, field=0, nest=GNILC_NEST, verbose=False)
    try:
        try:
            tau_err = hp.read_map(path, field=1, nest=GNILC_NEST)
        except TypeError:
            tau_err = hp.read_map(path, field=1, nest=GNILC_NEST, verbose=False)
    except Exception:
        tau_err = np.zeros_like(tau)
    nside = hp.npix2nside(len(tau))
    ebv = tau * TAU_TO_EBV
    ebv_err = tau_err * TAU_TO_EBV
    print(f"  GNILC opacity: nside={nside}, nest={GNILC_NEST}, "
          f"E(B-V) range {ebv.min():.4f} to {ebv.max():.4f} mag")
    return ebv, ebv_err, nside, GNILC_NEST


def query_ebv_at_stars(df: pd.DataFrame, ebv: np.ndarray, ebv_err: np.ndarray,
                       nside: int, nest: bool) -> pd.DataFrame:
    """Sample the E(B-V) map at each star's (ra, dec).

    Records the HEALPix pixel index too, so the pixel-overlap diagnostic can
    count how many stars share a pixel within and between samples.
    """
    ra = df["ra"].to_numpy(float)
    dec = df["dec"].to_numpy(float)
    ipix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
    df = df.copy()
    df["ebv"] = ebv[ipix]
    df["ebv_err"] = ebv_err[ipix]
    df["ipix"] = ipix
    return df


def pixel_overlap_diagnostic(df_all: pd.DataFrame, nside: int, out_path: str) -> None:
    """Diagnose how the two samples share GNILC pixels.

    """
    pix_robo = df_all.loc[df_all["sample"] == "RoboPol", "ipix"].to_numpy(int)
    pix_pano = df_all.loc[df_all["sample"] == "Panopoulou+2025", "ipix"].to_numpy(int)

    uniq_robo = np.unique(pix_robo)
    uniq_pano = np.unique(pix_pano)
    shared_pix = np.intersect1d(uniq_robo, uniq_pano)
    counts_robo = np.bincount(pix_robo - pix_robo.min(), minlength=1) if len(pix_robo) else np.array([])
    counts_pano = np.bincount(pix_pano - pix_pano.min(), minlength=1) if len(pix_pano) else np.array([])
    # Drop zero entries (most of the bincount range is empty).
    counts_robo = counts_robo[counts_robo > 0]
    counts_pano = counts_pano[counts_pano > 0]

    pix_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600.0
    pix_side_arcmin = np.sqrt(pix_area_arcmin2)
    print()
    print("GNILC pixel-overlap diagnostic:")
    print(f"  nside={nside}, pixel area ~ {pix_area_arcmin2:.2f} arcmin^2 "
          f"(~{pix_side_arcmin:.2f} arcmin/side, effective beam ~5 arcmin)")
    print(f"  RoboPol:        {len(pix_robo)} stars in {len(uniq_robo)} unique pixels "
          f"(mean {len(pix_robo)/max(len(uniq_robo),1):.2f} stars/pix)")
    print(f"  Panopoulou:     {len(pix_pano)} stars in {len(uniq_pano)} unique pixels "
          f"(mean {len(pix_pano)/max(len(uniq_pano),1):.2f} stars/pix)")
    print(f"  Pixels with both RoboPol AND Panopoulou stars: {len(shared_pix)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # Left: bar of stars-per-pixel for each sample.
    max_n = int(max(counts_robo.max() if len(counts_robo) else 1,
                    counts_pano.max() if len(counts_pano) else 1))
    bins = np.arange(0.5, max_n + 1.5, 1.0)
    ax = axes[0]
    ax.hist(counts_robo, bins=bins, color="tab:blue", alpha=0.6,
            edgecolor="black", lw=0.4, label=f"RoboPol ({len(pix_robo)} stars)")
    ax.hist(counts_pano, bins=bins, color="tab:purple", alpha=0.6,
            edgecolor="black", lw=0.4, label=f"Panopoulou+2025 ({len(pix_pano)} stars)")
    ax.set_xlabel("Stars per GNILC pixel")
    ax.set_ylabel("Number of pixels")
    ax.set_xticks(np.arange(1, max_n + 1))
    ax.legend(fontsize=8)
    ax.set_title("Stars per GNILC pixel, per sample")

    # Right: bar showing pixel-level overlap.
    ax = axes[1]
    cats = ["RoboPol only", "Panopoulou only", "Both"]
    vals = [
        int(len(np.setdiff1d(uniq_robo, uniq_pano))),
        int(len(np.setdiff1d(uniq_pano, uniq_robo))),
        int(len(shared_pix)),
    ]
    colors = ["tab:blue", "tab:purple", "tab:orange"]
    bars = ax.bar(cats, vals, color=colors, edgecolor="black", lw=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of pixels")
    ax.set_title("Pixel-level overlap between samples")
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) else 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote pixel-overlap figure: {out_path}")



def running_means(df: pd.DataFrame, n_per_bin: int) -> pd.DataFrame:
    """Equal-count binning in E(B-V); return median p with 5th/95th percentiles.

    Mirrors the binning style of Planck 2018 XII Fig. 12 and Fig. 25, which use
    equal numbers of points per bin in column density.
    """
    df = df.sort_values("ebv").reset_index(drop=True)
    n = len(df)
    if n < n_per_bin:
        return pd.DataFrame(columns=["ebv_mid", "p_p50", "p_p5", "p_p95", "n"])
    rows = []
    for start in range(0, n - n_per_bin + 1, n_per_bin):
        sub = df.iloc[start:start + n_per_bin]
        rows.append({
            "ebv_mid": sub["ebv"].median(),
            "p_p50": sub["p_percent"].median(),
            "p_p5": np.percentile(sub["p_percent"], 5),
            "p_p95": np.percentile(sub["p_percent"], 95),
            "n": len(sub),
        })
    return pd.DataFrame(rows)



def plot_panels(df_all: pd.DataFrame, run_robo: pd.DataFrame,
                run_pano: pd.DataFrame, run_pano_fg: pd.DataFrame,
                run_pano_bg: pd.DataFrame, out_path: str) -> None:
    df_robo = df_all[df_all["sample"] == "RoboPol"]
    df_pano = df_all[df_all["sample"] == "Panopoulou+2025"]
    df_pano_fg = df_pano[df_pano["distance_class"] == "foreground"]
    df_pano_bg = df_pano[df_pano["distance_class"] == "background"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), sharex=True, sharey=True)

    # Common log axes. Pick a range that covers both samples and a bit of margin.
    ebv_all = df_all["ebv"].to_numpy(float)
    p_all = df_all["p_percent"].to_numpy(float)
    pos = (ebv_all > 0) & (p_all > 0)
    if pos.any():
        ebv_lo = max(1e-3, np.percentile(ebv_all[pos], 1) * 0.5)
        ebv_hi = np.percentile(ebv_all[pos], 99.5) * 2.0
        p_lo = max(1e-2, np.percentile(p_all[pos], 1) * 0.5)
        p_hi = np.percentile(p_all[pos], 99.5) * 2.0
    else:
        ebv_lo, ebv_hi, p_lo, p_hi = 1e-3, 1.0, 1e-2, 10.0

    # Serkowski upper limit p_V <= 9% * E(B-V).
    ebv_line = np.geomspace(ebv_lo, ebv_hi, 100)
    serkowski = 9.0 * ebv_line  # percent

    def scatter(ax, df, color, label, alpha=0.6):
        ax.errorbar(df["ebv"], df["p_percent"],
                    yerr=df["sigma_p_percent"], xerr=None,
                    fmt="o", ms=3.0, mfc=color, mec="black", mew=0.3,
                    ecolor=color, alpha=alpha, lw=0.6, label=label)

    # Panel 1: overlay.
    ax = axes[0]
    scatter(ax, df_robo, "tab:blue", f"RoboPol (N={len(df_robo)})")
    scatter(ax, df_pano, "tab:purple", f"Panopoulou+2025 (N={len(df_pano)})")
    ax.plot(ebv_line, serkowski, "k--", lw=1.0, label=r"$p_V = 9\%\,E(B{-}V)$")
    if len(run_robo):
        ax.plot(run_robo["ebv_mid"], run_robo["p_p50"], color="tab:blue",
                lw=2.0, marker="s", ms=5, label="RoboPol median")
    if len(run_pano):
        ax.plot(run_pano["ebv_mid"], run_pano["p_p50"], color="tab:purple",
                lw=2.0, marker="s", ms=5, label="Panopoulou median")
    ax.set_title("Combined")
    ax.set_xlabel(r"$E(B{-}V)$ from Planck GNILC [mag]")
    ax.set_ylabel(r"$p_V$ [%]")
    ax.legend(fontsize=7, loc="lower right")

    # Panel 2: RoboPol alone.
    ax = axes[1]
    scatter(ax, df_robo, "tab:blue", f"RoboPol (N={len(df_robo)})")
    ax.plot(ebv_line, serkowski, "k--", lw=1.0)
    if len(run_robo):
        ax.fill_between(run_robo["ebv_mid"], run_robo["p_p5"], run_robo["p_p95"],
                        color="tab:blue", alpha=0.15, label="5-95%")
        ax.plot(run_robo["ebv_mid"], run_robo["p_p50"], color="tab:blue",
                lw=2.0, marker="s", ms=5, label="median")
    ax.set_title("RoboPol")
    ax.set_xlabel(r"$E(B{-}V)$ from Planck GNILC [mag]")
    ax.legend(fontsize=7, loc="lower right")

    # Panel 3: Panopoulou alone.
    ax = axes[2]
    scatter(ax, df_pano, "tab:purple", f"Panopoulou+2025 (N={len(df_pano)})")
    ax.plot(ebv_line, serkowski, "k--", lw=1.0)
    if len(run_pano):
        ax.fill_between(run_pano["ebv_mid"], run_pano["p_p5"], run_pano["p_p95"],
                        color="tab:purple", alpha=0.15, label="5-95%")
        ax.plot(run_pano["ebv_mid"], run_pano["p_p50"], color="tab:purple",
                lw=2.0, marker="s", ms=5, label="median")
    ax.set_title("Panopoulou+2025, all")
    ax.set_xlabel(r"$E(B{-}V)$ from Planck GNILC [mag]")
    ax.legend(fontsize=7, loc="lower right")

    # Panel 4: Panopoulou foreground vs background, with RoboPol median for reference.
    ax = axes[3]
    scatter(ax, df_pano_fg, "tab:orange",
            f"Pano. foreground d<{FOREGROUND_DIST_MAX_PC:.0f} pc (N={len(df_pano_fg)})",
            alpha=0.55)
    scatter(ax, df_pano_bg, "tab:green",
            f"Pano. background d>{BACKGROUND_DIST_MIN_PC:.0f} pc (N={len(df_pano_bg)})",
            alpha=0.55)
    ax.plot(ebv_line, serkowski, "k--", lw=1.0)
    if len(run_pano_fg):
        ax.plot(run_pano_fg["ebv_mid"], run_pano_fg["p_p50"], color="tab:orange",
                lw=2.0, marker="s", ms=5, label="foreground median")
    if len(run_pano_bg):
        ax.plot(run_pano_bg["ebv_mid"], run_pano_bg["p_p50"], color="tab:green",
                lw=2.0, marker="s", ms=5, label="background median")
    if len(run_robo):
        ax.plot(run_robo["ebv_mid"], run_robo["p_p50"], color="tab:blue",
                lw=1.5, ls=":", marker="o", ms=4, label="RoboPol median (ref.)")
    ax.set_title("Panopoulou+2025 split by distance")
    ax.set_xlabel(r"$E(B{-}V)$ from Planck GNILC [mag]")
    ax.legend(fontsize=7, loc="lower right")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ebv_lo, ebv_hi)
        ax.set_ylim(p_lo, p_hi)
        ax.grid(True, which="both", alpha=0.25, lw=0.4)

    fig.suptitle(
        r"Polarization fraction vs reddening (Markkanen region)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote figure: {out_path}")



def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading RoboPol...")
    df_robo = load_robopol(ROBOPOL_CSV)
    print("Loading Panopoulou+2025 polygon catalog...")
    df_pano = load_panopoulou(PANOPOULOU_CSV)

    print("Loading GNILC opacity map...")
    ebv_map, ebv_err_map, nside, nest = load_gnilc_ebv(GNILC_FITS)

    print("Sampling reddening at each star...")
    df_robo = query_ebv_at_stars(df_robo, ebv_map, ebv_err_map, nside, nest)
    df_pano = query_ebv_at_stars(df_pano, ebv_map, ebv_err_map, nside, nest)

    df_all = pd.concat([df_robo, df_pano], ignore_index=True)

    # Optional SNR cut.
    if MIN_PSNR is not None:
        snr = df_all["p_percent"] / df_all["sigma_p_percent"].replace(0, np.nan)
        before = len(df_all)
        df_all = df_all[snr >= MIN_PSNR].copy()
        print(f"  SNR cut p/sigma_p >= {MIN_PSNR}: kept {len(df_all)}/{before}")

    # Drop non-positive E(B-V) for log axes.
    df_all = df_all[df_all["ebv"] > 0].copy()

    df_all.to_csv(OUT_TABLE, index=False)
    print(f"  Wrote per-star table: {OUT_TABLE}")

    print("Computing running medians...")
    run_robo = running_means(df_all[df_all["sample"] == "RoboPol"], RUNNING_BIN_COUNT)
    run_pano = running_means(df_all[df_all["sample"] == "Panopoulou+2025"], RUNNING_BIN_COUNT)
    pano = df_all[df_all["sample"] == "Panopoulou+2025"]
    run_pano_fg = running_means(pano[pano["distance_class"] == "foreground"],
                                RUNNING_BIN_COUNT_SPLIT)
    run_pano_bg = running_means(pano[pano["distance_class"] == "background"],
                                RUNNING_BIN_COUNT_SPLIT)
    run_robo.insert(0, "sample", "RoboPol")
    run_pano.insert(0, "sample", "Panopoulou+2025")
    run_pano_fg.insert(0, "sample", "Panopoulou+2025_foreground")
    run_pano_bg.insert(0, "sample", "Panopoulou+2025_background")
    pd.concat([run_robo, run_pano, run_pano_fg, run_pano_bg],
              ignore_index=True).to_csv(OUT_RUNNING, index=False)
    print(f"  Wrote running-means table: {OUT_RUNNING}")

    print("Plotting...")
    plot_panels(df_all, run_robo, run_pano, run_pano_fg, run_pano_bg, OUT_FIG)

    # Pixel-overlap diagnostic.
    pixel_overlap_diagnostic(df_all, nside, OUT_PIXEL_HIST)

    # Quick text summary.
    print("\nSummary by sample:")
    for s in ["RoboPol", "Panopoulou+2025"]:
        sub = df_all[df_all["sample"] == s]
        if len(sub) == 0:
            continue
        print(f"  {s:>16}: N={len(sub):3d}  "
              f"<p>={sub['p_percent'].mean():.3f}%  "
              f"<E(B-V)>={sub['ebv'].mean():.4f} mag  "
              f"median p/E(B-V)={(sub['p_percent']/sub['ebv']).median():.2f} %/mag")

    print("\nSummary, Panopoulou by distance class:")
    pano = df_all[df_all["sample"] == "Panopoulou+2025"]
    for cls in ["foreground", "middle", "background", "unknown"]:
        sub = pano[pano["distance_class"] == cls]
        if len(sub) == 0:
            continue
        print(f"  {cls:>11}: N={len(sub):3d}  "
              f"median p={sub['p_percent'].median():.3f}%  "
              f"median E(B-V)={sub['ebv'].median():.4f} mag")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
pd_vs_reddening_zg.py

PD vs A0 (ZGR23 per-star extinction) for the Markkanen cloud analysis.
Mirrors the 4-panel layout of pd_vs_reddening.py but uses Zhang & Green (2023)
per-star A0 instead of GNILC E(B-V), avoiding the line-of-sight integration
bias. ZGR23 assumes R(V) = 3.1 throughout; no per-star R(V) is available from
this catalog.

Run from repo root:
    python coursework/6_reddening_maps/pd_vs_reddening_zg.py

TOPCAT query results (xpparams.main, one row per matched star) must be
produced separately via prepare_zgr23_ids.py + TOPCAT TAP query and placed
at the paths below.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# toggles 
QUALITY_FLAG_MAX   = 8        # 0 = cleanest only; 8 = ZGR23 recommended relaxed cut
ERR_EXT_MAX        = 0.2      # mag; drop stars with large A0 uncertainty
DIST_FOREGROUND_PC = 150      # Panopoulou foreground cut (d < this)
DIST_BACKGROUND_PC = 200      # Panopoulou background cut (d > this)
FILTER_ID          = 0        # Panopoulou FilterID to keep (0 = main sample)
SHOW_QUALITY_FAILS = True     # plot stars that fail ZGR23 quality cut as grey

# paths 
MERGED_OUTPUT      = "../2_sky_plot/merged_output.csv"
MERGED_WITH_GAIAID = "../4_gaia/match_with_gaia/merged_with_gaiaid.csv"
PANOPOULOU         = "../0_data/R/external_panopoulou_expanded_polygon.csv"
ZGR23_ROBOPOL      = "./zhang_green_map/xpparams_robopol_match.csv"
ZGR23_PANOPOULOU   = "./zhang_green_map/xpparams_panop_match.csv"
OUTPUT_DIR         = "./zhang_green_map"

# output filename suffix from active toggles
suffix_parts = [f"qflag{QUALITY_FLAG_MAX}", f"eerr{ERR_EXT_MAX}"]
if FILTER_ID is not None:
    suffix_parts.append(f"fid{FILTER_ID}")
SUFFIX = "_".join(suffix_parts)

#load data 
robopol_pol  = pd.read_csv(MERGED_OUTPUT)
gaia_ids     = pd.read_csv(MERGED_WITH_GAIAID)[["Name", "gaia_id"]].dropna(subset=["gaia_id"])
gaia_ids["gaia_id"] = gaia_ids["gaia_id"].astype("int64")
robopol_pol  = robopol_pol.merge(gaia_ids, on="Name", how="inner")
robopol_pol  = robopol_pol.rename(columns={"gaia_id": "source_id"})
robopol_pol["source_id"] = robopol_pol["source_id"].astype("int64")

pano_pol     = pd.read_csv(PANOPOULOU)
pano_pol     = pano_pol.rename(columns={"GID": "source_id"})
pano_pol["source_id"] = pano_pol["source_id"].astype("int64")

zgr_rob      = pd.read_csv(ZGR23_ROBOPOL)
zgr_rob["source_id"] = zgr_rob["source_id"].astype("int64")

zgr_pan      = pd.read_csv(ZGR23_PANOPOULOU)
zgr_pan["source_id"] = zgr_pan["source_id"].astype("int64")

# ── merge polarimetry with ZGR23 ───────────────────────────────────────────────
rob = robopol_pol.merge(zgr_rob, on="source_id", how="inner")
pan = pano_pol.merge(zgr_pan, on="source_id", how="inner")
# Panopoulou uses "p" for polarization degree; normalise to match RoboPol
if "p" in pan.columns and "P[%]" not in pan.columns:
    pan = pan.rename(columns={"p": "P[%]", "e_p": "sP[%]"})
    pan["P[%]"] = pan["P[%]"] * 100 # convert fraction to percent
    pan["sP[%]"] = pan["sP[%]"] * 100

print(f"RoboPol:     {len(robopol_pol)} stars with Gaia ID, "
      f"{len(rob)} matched in ZGR23")
print(f"Panopoulou:  {len(pano_pol)} stars in polygon, "
      f"{len(pan)} matched in ZGR23")

#apply FilterID cut to Panopoulou 
if FILTER_ID is not None:
    pan = pan[pan["FilterID"] == FILTER_ID]
    print(f"Panopoulou after FilterID={FILTER_ID}: {len(pan)} stars")

#  quality cuts 
def apply_quality(df, label):
    fail_flag = df["quality_flags"] > QUALITY_FLAG_MAX
    fail_err  = df["err_ext"] > ERR_EXT_MAX
    fail      = fail_flag | fail_err
    n_fail    = fail.sum()
    print(f"{label}: {n_fail} stars fail quality cut "
          f"(qflag>{QUALITY_FLAG_MAX} or err_ext>{ERR_EXT_MAX}), "
          f"{(~fail).sum()} pass")
    return df[~fail].copy(), df[fail].copy()

rob_good, rob_bad = apply_quality(rob, "RoboPol")
pan_good, pan_bad = apply_quality(pan, "Panopoulou")

# distance split for Panopoulou 
pan_fg = pan_good[pan_good["r_med_geo"] < DIST_FOREGROUND_PC].copy()
pan_bg = pan_good[pan_good["r_med_geo"] > DIST_BACKGROUND_PC].copy()
pan_mid = pan_good[
    (pan_good["r_med_geo"] >= DIST_FOREGROUND_PC) &
    (pan_good["r_med_geo"] <= DIST_BACKGROUND_PC)
].copy()

print(f"Panopoulou distance split: "
      f"fg (d<{DIST_FOREGROUND_PC}pc) N={len(pan_fg)}, "
      f"bg (d>{DIST_BACKGROUND_PC}pc) N={len(pan_bg)}, "
      f"mid N={len(pan_mid)} (excluded from split panel)")

#plotting helpers
ROB_COLOR = "#1f77b4"
PAN_COLOR = "#d62728"
FG_COLOR  = "#2ca02c"
BG_COLOR  = "#ff7f0e"
FAIL_COLOR = "#cccccc"

ALPHA_PT  = 0.5
PT_SIZE   = 18

def scatter_with_errorbars(ax, df, color, label, zorder=2):
    ax.errorbar(
        df["ext"], df["P[%]"],
        xerr=df["err_ext"],
        fmt="o", ms=4, color=color, ecolor=color,
        elinewidth=0.6, capsize=1.5, alpha=ALPHA_PT,
        label=label, zorder=zorder
    )

def add_median_line(ax, df, color, x_col="ext", y_col="P[%]", n_bins=5):
    """Bin by x, plot median y per bin."""
    if len(df) < 5:
        return
    df = df.copy().sort_values(x_col)
    bins = np.array_split(df, n_bins)
    xs = [b[x_col].median() for b in bins if len(b) > 0]
    ys = [b[y_col].median() for b in bins if len(b) > 0]
    ax.plot(xs, ys, "-", color=color, lw=1.5, alpha=0.8, zorder=3)

def format_ax(ax):
    ax.set_xlabel(r"$A_0$ (ZGR23, mag)", fontsize=10)
    ax.set_ylabel(r"$p$ (%)", fontsize=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)

#figure: 4 panels
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
ax_all, ax_rob, ax_pan, ax_dist = axes.flat

# panel 1: combined
if SHOW_QUALITY_FAILS:
    scatter_with_errorbars(ax_all, rob_bad, FAIL_COLOR, "_nolegend_", zorder=1)
    scatter_with_errorbars(ax_all, pan_bad, FAIL_COLOR, "_nolegend_", zorder=1)
scatter_with_errorbars(ax_all, rob_good, ROB_COLOR,
                       f"RoboPol (N={len(rob_good)})")
scatter_with_errorbars(ax_all, pan_good, PAN_COLOR,
                       f"Panopoulou+2025 (N={len(pan_good)})")
add_median_line(ax_all, rob_good, ROB_COLOR)
add_median_line(ax_all, pan_good, PAN_COLOR)
ax_all.set_title("Combined", fontsize=11)
ax_all.legend(fontsize=8, framealpha=0.7)
format_ax(ax_all)

# panel 2: RoboPol only
if SHOW_QUALITY_FAILS:
    scatter_with_errorbars(ax_rob, rob_bad, FAIL_COLOR,
                           f"quality fail (N={len(rob_bad)})", zorder=1)
scatter_with_errorbars(ax_rob, rob_good, ROB_COLOR,
                       f"RoboPol (N={len(rob_good)})")
add_median_line(ax_rob, rob_good, ROB_COLOR)
ax_rob.set_title("RoboPol", fontsize=11)
ax_rob.legend(fontsize=8, framealpha=0.7)
format_ax(ax_rob)

# panel 3: Panopoulou only
if SHOW_QUALITY_FAILS:
    scatter_with_errorbars(ax_pan, pan_bad, FAIL_COLOR,
                           f"quality fail (N={len(pan_bad)})", zorder=1)
scatter_with_errorbars(ax_pan, pan_good, PAN_COLOR,
                       f"Panopoulou+2025 (N={len(pan_good)})")
add_median_line(ax_pan, pan_good, PAN_COLOR)
ax_pan.set_title("Panopoulou+2025", fontsize=11)
ax_pan.legend(fontsize=8, framealpha=0.7)
format_ax(ax_pan)

# panel 4: Panopoulou split by distance
scatter_with_errorbars(ax_dist, pan_fg, FG_COLOR,
                       f"fg d<{DIST_FOREGROUND_PC} pc (N={len(pan_fg)})")
scatter_with_errorbars(ax_dist, pan_bg, BG_COLOR,
                       f"bg d>{DIST_BACKGROUND_PC} pc (N={len(pan_bg)})")
add_median_line(ax_dist, pan_fg, FG_COLOR)
add_median_line(ax_dist, pan_bg, BG_COLOR)
ax_dist.set_title("Panopoulou+2025 by distance", fontsize=11)
ax_dist.legend(fontsize=8, framealpha=0.7)
format_ax(ax_dist)

fig.suptitle(
    f"$p$ vs $A_0$ (ZGR23, R(V)=3.1 assumed)  |  "
    f"quality_flags $\\leq$ {QUALITY_FLAG_MAX}, err_ext $<$ {ERR_EXT_MAX} mag",
    fontsize=11
)

outpath = os.path.join(OUTPUT_DIR, f"pd_vs_reddening_zgr23_{SUFFIX}.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nSaved: {outpath}")
plt.show()

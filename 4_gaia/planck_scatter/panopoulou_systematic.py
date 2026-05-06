#!/usr/bin/env python3
"""
panopoulou_systematic.py

Diagnostic of the q/u systematic offset between RoboPol and Panopoulou+25
on the Markkanen q/u-vs-distance plot ("all violet points
sit systematically below the blue ones").


Inputs:
    - merged_output.csv                   RoboPol q,u (equatorial IAU)
    - external_panopoulou_expanded_polygon.csv  Panopoulou+25 polygon catalog

Outputs (in OUT_DIR):
    - panopoulou_systematic_summary.txt   stdout dump in a file
    - panopoulou_systematic_by_filter.png small bar chart of <q>,<u> per FilterID

Run:
    python panopoulou_systematic.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sstats


# CONFIG  -- edit these paths to match your setup

ROBOPOL_CSV = os.path.expanduser(
    "~/Desktop/coursework/2_sky_plot/merged_output.csv"
)

PANOPOULOU_CSV = os.path.expanduser(
    "~/Desktop/coursework/0_data/R/external_panopoulou_expanded_polygon.csv"
)

OUT_DIR = os.path.expanduser(
    "~/Desktop/coursework/4_gaia/planck_scatter/output_panopoulou_diagnostic"
)

FIG_DPI = 220

# Bailer-Jones distances for RoboPol stars (joined on Name).
# Used by section 4 localized distance-window comparisons.
BJ_DISTANCES_CSV = os.path.expanduser(
    "~/Desktop/coursework/4_gaia/bj_distances.csv"
)

# Section 4: localized distance-window t-tests
# Toggle on/off and pick windows. Each window is (d_min_pc, d_max_pc, label).
RUN_DISTANCE_TTEST = True
DISTANCE_WINDOWS = [
    (150.0, 400.0, "150-400 pc (primary overlap)"),
    (0.0,   200.0, "0-200 pc (foreground)"),
    (200.0, 500.0, "200-500 pc (cloud regime)"),
]


def read_table_auto(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_robopol(path):
    df = read_table_auto(path)
    needed = {"Name", "q", "sq", "u", "su"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {missing}")
    if np.nanmax(np.abs(df[["q", "u"]].to_numpy(float))) < 0.1:
        for c in ("q", "u", "sq", "su"):
            df[c] = df[c] * 100.0
        print(f"  RoboPol: {len(df)} stars, q/u promoted to percent")
    else:
        print(f"  RoboPol: {len(df)} stars, q/u already in percent")
    return df


def load_panopoulou(path):
    """Read Panopoulou+25 polygon catalog and build q,u in percent."""
    cat = read_table_auto(path)
    needed = ["RA", "DEC", "p", "e_p", "evpa", "e_evpa", "FilterID"]
    missing = [c for c in needed if c not in cat.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {missing}")

    for c in ["p", "e_p", "evpa", "e_evpa"]:
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    cat = cat.dropna(subset=["p", "evpa", "e_p", "e_evpa"]).copy()

    # Panopoulou stores p as fractional, convert to percent.
    cat["p_pct"] = cat["p"] * 100.0
    cat["ep_pct"] = cat["e_p"] * 100.0

    pa_rad = np.radians(cat["evpa"].to_numpy(float))
    cat["q"] = cat["p_pct"] * np.cos(2.0 * pa_rad)
    cat["u"] = cat["p_pct"] * np.sin(2.0 * pa_rad)

    pa_err_rad = np.radians(cat["e_evpa"].to_numpy(float))
    cos2t = np.cos(2.0 * pa_rad)
    sin2t = np.sin(2.0 * pa_rad)
    cat["sq"] = np.sqrt((cos2t * cat["ep_pct"]) ** 2
                        + (2.0 * cat["p_pct"] * sin2t * pa_err_rad) ** 2)
    cat["su"] = np.sqrt((sin2t * cat["ep_pct"]) ** 2
                        + (2.0 * cat["p_pct"] * cos2t * pa_err_rad) ** 2)

    print(f"  Panopoulou: {len(cat)} stars (p,evpa parsed)")
    return cat



# Statistics

def wmean(values, errors):
    v = np.asarray(values, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    if ok.sum() == 0:
        return np.nan, np.nan, 0
    w = 1.0 / e[ok] ** 2
    m = (w * v[ok]).sum() / w.sum()
    se = np.sqrt(1.0 / w.sum())
    return float(m), float(se), int(ok.sum())

# Distance merging for the localized window comparison

def attach_robopol_distances(df_robopol, bj_path):
    """Inner-join RoboPol q,u table with Bailer-Jones distances on Name.

    Returns a frame with an added r_med_photogeo column. Stars without a
    distance match are dropped silently. If the BJ file is missing, returns
    None and the caller should skip the distance-window section.
    """
    if not os.path.exists(bj_path):
        print(f"  WARNING: BJ distance file not found: {bj_path}")
        print("  Section [4] (distance-window t-tests) will be skipped.")
        return None
    bj = read_table_auto(bj_path)
    if "Name" not in bj.columns or "r_med_photogeo" not in bj.columns:
        print(f"  WARNING: {bj_path} missing 'Name' or 'r_med_photogeo'.")
        print("  Section [4] (distance-window t-tests) will be skipped.")
        return None
    bj = bj[["Name", "r_med_photogeo"]].copy()
    bj["r_med_photogeo"] = pd.to_numeric(bj["r_med_photogeo"], errors="coerce")
    out = df_robopol.merge(bj, on="Name", how="inner")
    out = out.dropna(subset=["r_med_photogeo"]).copy()
    print(f"  RoboPol with BJ distances: {len(out)}/{len(df_robopol)} stars matched")
    return out


def attach_panopoulou_distances(df_pano):
    """Ensure r_med_photogeo is numeric and present, drop rows missing it.

    The polygon catalog already carries Bailer-Jones distances,
    """
    if "r_med_photogeo" not in df_pano.columns:
        print("  WARNING: Panopoulou frame has no r_med_photogeo column.")
        print("  Section [4] (distance-window t-tests) will be skipped.")
        return None
    out = df_pano.copy()
    out["r_med_photogeo"] = pd.to_numeric(out["r_med_photogeo"], errors="coerce")
    out = out.dropna(subset=["r_med_photogeo"]).copy()
    return out


# Three-flavour mean-difference test
# - z-test on weighted means (the test given heteroscedastic sigma)
# - Student's t-test (equal variance, unweighted)
# - Welch's t-test (unequal variance, unweighted)
# All three are returned so we can report them side by side.

def compare_means_three_ways(values_a, errors_a, values_b, errors_b):
    """Compare two samples on a single Stokes parameter.

    Returns a dict with weighted means, weighted-mean z statistic, and
    Student's/Welch's t-statistics with their two-sided p-values.

    """
    a = np.asarray(values_a, dtype=float)
    ea = np.asarray(errors_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    eb = np.asarray(errors_b, dtype=float)

    oka = np.isfinite(a) & np.isfinite(ea) & (ea > 0)
    okb = np.isfinite(b) & np.isfinite(eb) & (eb > 0)
    a, ea = a[oka], ea[oka]
    b, eb = b[okb], eb[okb]

    out = {
        "n_a": int(a.size), "n_b": int(b.size),
        "wmean_a": np.nan, "werr_a": np.nan,
        "wmean_b": np.nan, "werr_b": np.nan,
        "delta": np.nan, "sdelta": np.nan, "z": np.nan, "p_z": np.nan,
        "t_student": np.nan, "dof_student": np.nan, "p_student": np.nan,
        "t_welch": np.nan, "dof_welch": np.nan, "p_welch": np.nan,
    }
    if a.size < 2 or b.size < 2:
        return out

    # weighted means with 1/sigma^2 weights
    wa = 1.0 / ea ** 2
    wb = 1.0 / eb ** 2
    ma = (wa * a).sum() / wa.sum()
    mb = (wb * b).sum() / wb.sum()
    sa = np.sqrt(1.0 / wa.sum())
    sb = np.sqrt(1.0 / wb.sum())
    delta = ma - mb
    sdelta = np.sqrt(sa ** 2 + sb ** 2)
    z = delta / sdelta if sdelta > 0 else np.nan
    p_z = 2.0 * sstats.norm.sf(abs(z)) if np.isfinite(z) else np.nan

    out["wmean_a"], out["werr_a"] = float(ma), float(sa)
    out["wmean_b"], out["werr_b"] = float(mb), float(sb)
    out["delta"], out["sdelta"] = float(delta), float(sdelta)
    out["z"], out["p_z"] = float(z), float(p_z)

    # Student's t-test (equal variance, unweighted)
    t_s, p_s = sstats.ttest_ind(a, b, equal_var=True)
    out["t_student"] = float(t_s)
    out["dof_student"] = float(a.size + b.size - 2)
    out["p_student"] = float(p_s)

    # Welch's t-test (unequal variance, unweighted)
    t_w, p_w = sstats.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite dof
    va = a.var(ddof=1) if a.size > 1 else np.nan
    vb = b.var(ddof=1) if b.size > 1 else np.nan
    if a.size > 1 and b.size > 1 and va > 0 and vb > 0:
        num = (va / a.size + vb / b.size) ** 2
        den = (va ** 2) / (a.size ** 2 * (a.size - 1)) + \
              (vb ** 2) / (b.size ** 2 * (b.size - 1))
        dof_w = num / den
    else:
        dof_w = np.nan
    out["t_welch"] = float(t_w)
    out["dof_welch"] = float(dof_w)
    out["p_welch"] = float(p_w)
    return out



def report(df_pano, df_robopol, out_txt):
    lines = []
    P = lambda s: lines.append(s)

    P("=" * 70)
    P("Panopoulou+25 systematic diagnostic (Markkanen field)")
    P("=" * 70)

    # --- FilterID breakdown inside Panopoulou ---
    P("\n[1] Panopoulou: weighted means by FilterID")
    P(f"{'FilterID':>10} {'N':>6} {'<q>':>10} {'sigma':>10} "
      f"{'<u>':>10} {'sigma':>10}")
    for fid, grp in df_pano.groupby("FilterID", dropna=False):
        qm, qe, _ = wmean(grp["q"], grp["sq"])
        um, ue, _ = wmean(grp["u"], grp["su"])
        P(f"{str(fid):>10} {len(grp):>6} "
          f"{qm:>+10.4f} {qe:>10.4f} "
          f"{um:>+10.4f} {ue:>10.4f}")

    # --- Other groupings if present ---
    other_cols = [c for c in df_pano.columns
                  if c.lower() in ("origin", "project", "cat", "survey", "ref")]
    for col in other_cols:
        P(f"\n[1b] Panopoulou: weighted means by {col}")
        P(f"{col:>16} {'N':>6} {'<q>':>10} {'sigma':>10} "
          f"{'<u>':>10} {'sigma':>10}")
        for v, grp in df_pano.groupby(col, dropna=False):
            qm, qe, _ = wmean(grp["q"], grp["sq"])
            um, ue, _ = wmean(grp["u"], grp["su"])
            P(f"{str(v):>16} {len(grp):>6} "
              f"{qm:>+10.4f} {qe:>10.4f} "
              f"{um:>+10.4f} {ue:>10.4f}")

    # --- Whole-sample comparison ---
    qm_p, qe_p, np_ = wmean(df_pano["q"], df_pano["sq"])
    um_p, ue_p, _ = wmean(df_pano["u"], df_pano["su"])
    qm_r, qe_r, nr_ = wmean(df_robopol["q"], df_robopol["sq"])
    um_r, ue_r, _ = wmean(df_robopol["u"], df_robopol["su"])

    P("\n[2] Whole-sample comparison")
    P(f"  Panopoulou+25 (N={np_}):  <q>={qm_p:+.4f}\u00b1{qe_p:.4f}  "
      f"<u>={um_p:+.4f}\u00b1{ue_p:.4f}")
    P(f"  RoboPol       (N={nr_}):  <q>={qm_r:+.4f}\u00b1{qe_r:.4f}  "
      f"<u>={um_r:+.4f}\u00b1{ue_r:.4f}")

    dq = qm_p - qm_r
    sdq = np.sqrt(qe_p ** 2 + qe_r ** 2)
    du = um_p - um_r
    sdu = np.sqrt(ue_p ** 2 + ue_r ** 2)
    P(f"\n[3] Pano - RoboPol offset")
    P(f"  delta q = {dq:+.4f} \u00b1 {sdq:.4f}  "
      f"({abs(dq) / sdq:.1f}\u03c3)")
    P(f"  delta u = {du:+.4f} \u00b1 {sdu:.4f}  "
      f"({abs(du) / sdu:.1f}\u03c3)")
    
    # --- Section [4]: localized distance-window comparisons ---
    if RUN_DISTANCE_TTEST:
        P("\n[4] Localized distance-window comparisons (Pano vs RoboPol)")
        if "r_med_photogeo" not in df_pano.columns or \
           "r_med_photogeo" not in df_robopol.columns:
            P("  Skipped: one of the samples has no r_med_photogeo column.")
        else:
            for d_lo, d_hi, label in DISTANCE_WINDOWS:
                mask_p = (df_pano["r_med_photogeo"] >= d_lo) & \
                         (df_pano["r_med_photogeo"] < d_hi)
                mask_r = (df_robopol["r_med_photogeo"] >= d_lo) & \
                         (df_robopol["r_med_photogeo"] < d_hi)
                gp = df_pano[mask_p]
                gr = df_robopol[mask_r]

                P(f"\n  Window: {label}  (Pano N={len(gp)}, RoboPol N={len(gr)})")
                if len(gp) < 2 or len(gr) < 2:
                    P("    Skipped: too few stars in this window.")
                    continue

                for stokes in ("q", "u"):
                    res = compare_means_three_ways(
                        gp[stokes], gp[f"s{stokes}"],
                        gr[stokes], gr[f"s{stokes}"],
                    )
                    P(f"    {stokes}:  "
                      f"<{stokes}>_pano={res['wmean_a']:+.4f}\u00b1{res['werr_a']:.4f}  "
                      f"<{stokes}>_robo={res['wmean_b']:+.4f}\u00b1{res['werr_b']:.4f}")
                    P(f"        delta = {res['delta']:+.4f} \u00b1 {res['sdelta']:.4f}  "
                      f"(weighted-mean z = {res['z']:+.2f}, p = {res['p_z']:.2e})")
                    P(f"        Student t = {res['t_student']:+.2f}  "
                      f"(dof={res['dof_student']:.0f}, p = {res['p_student']:.2e})")
                    P(f"        Welch   t = {res['t_welch']:+.2f}  "
                      f"(dof={res['dof_welch']:.1f}, p = {res['p_welch']:.2e})")
    
    
    P("=" * 70)

    txt = "\n".join(lines)
    print(txt)
    with open(out_txt, "w") as f:
        f.write(txt + "\n")



# Plot: per-FilterID weighted means

def plot_by_filter(df_pano, df_robopol, outfile):
    groups = []
    for fid, grp in df_pano.groupby("FilterID", dropna=False):
        qm, qe, _ = wmean(grp["q"], grp["sq"])
        um, ue, _ = wmean(grp["u"], grp["su"])
        groups.append((f"Pano F{fid} (N={len(grp)})", qm, qe, um, ue))
    qm, qe, _ = wmean(df_robopol["q"], df_robopol["sq"])
    um, ue, _ = wmean(df_robopol["u"], df_robopol["su"])
    groups.append((f"RoboPol (N={len(df_robopol)})", qm, qe, um, ue))

    labels = [g[0] for g in groups]
    qs = np.array([g[1] for g in groups])
    qe = np.array([g[2] for g in groups])
    us = np.array([g[3] for g in groups])
    ue = np.array([g[4] for g in groups])

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(groups)), 5))
    ax.errorbar(x - 0.12, qs, yerr=qe, fmt="o", ms=8, color="tab:orange",
                ecolor="tab:orange", capsize=3, label="<q>  [%]")
    ax.errorbar(x + 0.12, us, yerr=ue, fmt="s", ms=8, color="tab:purple",
                ecolor="tab:purple", capsize=3, label="<u>  [%]")
    ax.axhline(0, color="k", lw=0.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Weighted mean Stokes parameter  [%]")
    ax.set_title("Panopoulou+25 vs RoboPol: <q>, <u> per group")
    ax.legend(loc="best")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(outfile, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)



def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Panopoulou+25 vs RoboPol systematic diagnostic")
    print("=" * 70)

    print(f"\nLoading RoboPol from {ROBOPOL_CSV}")
    df_robopol = load_robopol(ROBOPOL_CSV)

    print(f"\nLoading Panopoulou from {PANOPOULOU_CSV}")
    df_pano = load_panopoulou(PANOPOULOU_CSV)

    if RUN_DISTANCE_TTEST:
        print("\nAttaching Bailer-Jones distances for distance-window section [4]")
        #df_robopol = attach_robopol_distances(df_robopol, BJ_DISTANCES_CSV) or df_robopol
        df_with_dist = attach_robopol_distances(df_robopol, BJ_DISTANCES_CSV)
        if df_with_dist is not None:
            df_robopol = df_with_dist
        #df_pano = attach_panopoulou_distances(df_pano) or df_pano
        df_with_dist_pano = attach_panopoulou_distances(df_pano)
        if df_with_dist_pano is not None:
            df_pano = df_with_dist_pano

    out_txt = os.path.join(OUT_DIR, "panopoulou_systematic_summary.txt")
    out_png = os.path.join(OUT_DIR, "panopoulou_systematic_by_filter.png")

    report(df_pano, df_robopol, out_txt)
    plot_by_filter(df_pano, df_robopol, out_png)
    

    print(f"\nWrote: {out_txt}")
    print(f"Wrote: {out_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()

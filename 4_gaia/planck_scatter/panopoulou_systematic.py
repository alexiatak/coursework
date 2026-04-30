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

    out_txt = os.path.join(OUT_DIR, "panopoulou_systematic_summary.txt")
    out_png = os.path.join(OUT_DIR, "panopoulou_systematic_by_filter.png")

    report(df_pano, df_robopol, out_txt)
    plot_by_filter(df_pano, df_robopol, out_png)

    print(f"\nWrote: {out_txt}")
    print(f"Wrote: {out_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()

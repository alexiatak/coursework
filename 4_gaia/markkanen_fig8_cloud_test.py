#!/usr/bin/env python3
"""

Fig. 8-style TRIShUL (https://arxiv.org/pdf/2510.25911) diagnostic for the Markkanen cloud.

This script is intentionally NOT the full TRIShUL pipeline. It makes the
paper-style q/u vs distance-modulus plot and runs some checks:

1. optional Bailer-Jones distance-quality cut
2. manual cloud-distance tests, e.g. 350, 400, 430, 500 pc
3. a simple automatic split test using q/u weighted means
4. before/after weighted q,u means and jump significances
5. a cumulative Mahalanobis diagnostic plot.

Edit only the paths and parameters in the CONFIG section, then run:

    python markkanen_fig8_cloud_test.py

Expected merged-output columns:
    Name, JD, P[%], sP[%], PA[deg], sPA[deg], q, sq, u, su, ra, dec

Expected distance-table columns:
    Name, gid, r_med_photogeo, r_lo_photogeo, r_hi_photogeo
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG: change these manually
# =============================================================================
MERGED_CSV = r"../2_sky_plot/merged_output.csv"      # your RoboPol merged table
BJ_CSV = r"bj_distances.csv"           # your Bailer-Jones distance table

OUT_PREPARED = r"markkanen_fig8_prepared.csv"
OUT_TESTS = r"markkanen_cloud_distance_tests.csv"
OUT_FIG = r"markkanen_fig8_with_cloud_test.png"
OUT_DIAG = r"markkanen_cumulative_mahalanobis.png"
OUT_DIR = "markkanen_fig8_distance_tests"
os.makedirs(OUT_DIR, exist_ok=True)




# Recommended first-pass cut. Set to None if you want to keep everything.
RELATIVE_DIST_ERR_MAX = 0.25


# Manual distances to test. Add/remove values here.
MANUAL_CLOUD_DISTANCES_PC = [300, 350, 400, 430, 450, 500, 600]
#MANUAL_CLOUD_DISTANCES_PC = [430]


# Automatic grid-search range for a possible step in q/u.
AUTO_GRID_MIN_PC = 200
AUTO_GRID_MAX_PC = 1500
AUTO_GRID_STEP_PC = 10
MIN_STARS_PER_SIDE = 8

FIG_DPI = 220



def read_table_auto(path: str) -> pd.DataFrame:
    """Read comma-, tab-, or whitespace-separated CSV-like table."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_and_prepare() -> pd.DataFrame:
    pol = read_table_auto(MERGED_CSV)
    bj = read_table_auto(BJ_CSV)

    required_pol = ["Name", "q", "sq", "u", "su", "ra", "dec"]
    required_bj = ["Name", "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]
    missing_pol = [c for c in required_pol if c not in pol.columns]
    missing_bj = [c for c in required_bj if c not in bj.columns]
    if missing_pol:
        raise ValueError(f"Missing columns in {MERGED_CSV}: {missing_pol}")
    if missing_bj:
        raise ValueError(f"Missing columns in {BJ_CSV}: {missing_bj}")

    keep_bj = ["Name", "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]
    if "gid" in bj.columns:
        keep_bj.insert(1, "gid")
    bj = bj[keep_bj].copy()

    df = pol.merge(bj, on="Name", how="inner")
    print(f"Polarization stars: {len(pol)}")
    print(f"Distance stars:     {len(bj)}")
    print(f"Joined stars:       {len(df)}")

    # Convert numeric columns safely.
    numeric_cols = ["q", "sq", "u", "su", "ra", "dec",
                    "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=numeric_cols).copy()

    #  q/u are fractional: Mark_0 q=-0.01185, u=0.01401 gives P=1.83%.
    # Convert to percent for TRIShUL-style plotting.
    if max(df["q"].abs().max(), df["u"].abs().max()) < 0.1:
        for c in ["q", "sq", "u", "su"]:
            df[c] *= 100.0
        print("Converted q, u, sq, su from fractional units to percent.")
    else:
        print("q/u look already in percent; no unit conversion applied.")

    # Distance modulus and uncertainty.
    d = df["r_med_photogeo"].to_numpy(float)
    d_lo = df["r_lo_photogeo"].to_numpy(float)
    d_hi = df["r_hi_photogeo"].to_numpy(float)
    df["mu"] = 5.0 * np.log10(d) - 5.0
    df["sigma_d"] = 0.5 * (d_hi - d_lo)
    df["rel_sigma_d"] = df["sigma_d"] / df["r_med_photogeo"]
    df["s_mu"] = (5.0 / np.log(10.0)) * df["rel_sigma_d"]

    if RELATIVE_DIST_ERR_MAX is not None:
        n0 = len(df)
        df = df[df["rel_sigma_d"] <= RELATIVE_DIST_ERR_MAX].copy()
        print(f"Distance-quality cut sigma_d/d <= {RELATIVE_DIST_ERR_MAX:.0%}: "
              f"kept {len(df)}/{n0} stars")

    df = df.sort_values("r_med_photogeo").reset_index(drop=True)
    return df


def weighted_mean_and_error(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    ok = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if ok.sum() == 0:
        return np.nan, np.nan
    w = 1.0 / errors[ok] ** 2
    mean = np.sum(w * values[ok]) / np.sum(w)
    err = np.sqrt(1.0 / np.sum(w))
    return float(mean), float(err)


def split_statistics(df: pd.DataFrame, cloud_pc: float) -> dict:
    before = df["r_med_photogeo"] < cloud_pc
    after = ~before
    nb = int(before.sum())
    na = int(after.sum())

    row = {
        "cloud_pc": cloud_pc,
        "cloud_mu": 5.0 * np.log10(cloud_pc) - 5.0,
        "n_before": nb,
        "n_after": na,
    }
    if nb < MIN_STARS_PER_SIDE or na < MIN_STARS_PER_SIDE:
        row.update({
            "q_before": np.nan, "eq_before": np.nan,
            "u_before": np.nan, "eu_before": np.nan,
            "q_after": np.nan, "eq_after": np.nan,
            "u_after": np.nan, "eu_after": np.nan,
            "delta_q": np.nan, "e_delta_q": np.nan, "sig_q": np.nan,
            "delta_u": np.nan, "e_delta_u": np.nan, "sig_u": np.nan,
            "jump_amp": np.nan, "jump_sig_diag": np.nan,
            "chi2_piecewise": np.nan,
        })
        return row

    qb, eqb = weighted_mean_and_error(df.loc[before, "q"], df.loc[before, "sq"])
    ub, eub = weighted_mean_and_error(df.loc[before, "u"], df.loc[before, "su"])
    qa, eqa = weighted_mean_and_error(df.loc[after, "q"], df.loc[after, "sq"])
    ua, eua = weighted_mean_and_error(df.loc[after, "u"], df.loc[after, "su"])

    dq = qa - qb
    du = ua - ub
    edq = np.sqrt(eqa**2 + eqb**2)
    edu = np.sqrt(eua**2 + eub**2)
    sig_q = dq / edq if edq > 0 else np.nan
    sig_u = du / edu if edu > 0 else np.nan
    jump_amp = np.sqrt(dq**2 + du**2)
    jump_sig_diag = np.sqrt(sig_q**2 + sig_u**2)

    # Weighted piecewise chi2 in q and u. Smaller = cleaner two-level step.
    chi2 = 0.0
    chi2 += np.sum(((df.loc[before, "q"] - qb) / df.loc[before, "sq"]) ** 2)
    chi2 += np.sum(((df.loc[before, "u"] - ub) / df.loc[before, "su"]) ** 2)
    chi2 += np.sum(((df.loc[after, "q"] - qa) / df.loc[after, "sq"]) ** 2)
    chi2 += np.sum(((df.loc[after, "u"] - ua) / df.loc[after, "su"]) ** 2)

    row.update({
        "q_before": qb, "eq_before": eqb,
        "u_before": ub, "eu_before": eub,
        "q_after": qa, "eq_after": eqa,
        "u_after": ua, "eu_after": eua,
        "delta_q": dq, "e_delta_q": edq, "sig_q": sig_q,
        "delta_u": du, "e_delta_u": edu, "sig_u": sig_u,
        "jump_amp": jump_amp,
        "jump_sig_diag": jump_sig_diag,
        "chi2_piecewise": float(chi2),
    })
    return row


def run_cloud_tests(df: pd.DataFrame) -> pd.DataFrame:
    manual = list(MANUAL_CLOUD_DISTANCES_PC)
    grid = list(np.arange(AUTO_GRID_MIN_PC, AUTO_GRID_MAX_PC + AUTO_GRID_STEP_PC,
                          AUTO_GRID_STEP_PC, dtype=float))
    all_d = sorted(set([float(x) for x in manual + grid]))
    tests = pd.DataFrame([split_statistics(df, d) for d in all_d])
    tests["is_manual"] = tests["cloud_pc"].isin([float(x) for x in manual])
    tests = tests.sort_values("cloud_pc").reset_index(drop=True)
    return tests


def choose_best_cloud(tests: pd.DataFrame) -> float | None:
    valid = tests.dropna(subset=["chi2_piecewise", "jump_sig_diag"]).copy()
    if valid.empty:
        return None
    # The jump significance is printed as
    # a sanity check, not used alone because it can favor arbitrary splits.
    best = valid.loc[valid["chi2_piecewise"].idxmin()]
    return float(best["cloud_pc"])


def cumulative_mahalanobis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("r_med_photogeo").reset_index(drop=True)
    q = out["q"].to_numpy(float)
    u = out["u"].to_numpy(float)
    sq = out["sq"].to_numpy(float)
    su = out["su"].to_numpy(float)
    out["dMaha"] = np.sqrt((q / sq) ** 2 + (u / su) ** 2)
    out["cum_dMaha"] = np.cumsum(out["dMaha"])
    return out


def simple_piecewise_linear_break(y: np.ndarray, min_size: int = 8) -> int | None:
    """Quick diagnostic breakpoint in cumulative dMaha, not full TRIShUL."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * min_size + 1:
        return None
    x = np.arange(n)
    best_i, best_ssr = None, np.inf
    for i in range(min_size, n - min_size):
        ssr = 0.0
        for sl in [slice(0, i), slice(i, n)]:
            xx = x[sl]
            yy = y[sl]
            p = np.polyfit(xx, yy, 1)
            ssr += np.sum((yy - np.polyval(p, xx)) ** 2)
        if ssr < best_ssr:
            best_ssr = ssr
            best_i = i
    return best_i


def plot_main(df: pd.DataFrame, tests: pd.DataFrame, cloud_pc, outfile=None):
#def plot_main(df: pd.DataFrame, tests: pd.DataFrame, cloud_pc: float | None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(df["mu"], df["q"], xerr=df["s_mu"], yerr=df["sq"],
                fmt="+", ms=5, color="tab:green", ecolor="tab:green",
                alpha=0.65, lw=0.8, capsize=0, label="q")
    ax.errorbar(df["mu"], df["u"], xerr=df["s_mu"], yerr=df["su"],
                fmt="+", ms=5, color="tab:blue", ecolor="tab:blue",
                alpha=0.65, lw=0.8, capsize=0, label="u")
    ax.axhline(0, color="k", lw=0.6, alpha=0.7)

    # Thin grey manual candidates.
    for d in MANUAL_CLOUD_DISTANCES_PC:
        if df["r_med_photogeo"].min() < d < df["r_med_photogeo"].max():
            ax.axvline(5 * np.log10(d) - 5, color="0.5", lw=0.7,
                       alpha=0.25, ls="--")

    if cloud_pc is not None:
        st = split_statistics(df, cloud_pc)
        mu_c = st["cloud_mu"]
        before = df["r_med_photogeo"] < cloud_pc
        after = ~before
        ax.axvline(mu_c, color="purple", lw=1.4, alpha=0.8,
                   label=f"tested cloud: {cloud_pc:.0f} pc")
        ymin, ymax = ax.get_ylim()
        ax.text(mu_c, ymax - 0.04 * (ymax - ymin), f"  {cloud_pc:.0f} pc",
                color="purple", va="top", ha="left", fontsize=9)

        if before.any():
            xmin = df.loc[before, "mu"].min()
            ax.hlines(st["q_before"], xmin, mu_c, colors="tab:green", lw=2.0)
            ax.hlines(st["u_before"], xmin, mu_c, colors="tab:blue", lw=2.0)
        if after.any():
            xmax = df.loc[after, "mu"].max()
            ax.hlines(st["q_after"], mu_c, xmax, colors="tab:green", lw=2.0)
            ax.hlines(st["u_after"], mu_c, xmax, colors="tab:blue", lw=2.0)

    ax.set_xlabel(r"Distance modulus  $\mu$")
    ax.set_ylabel("Stokes parameters  [%]")
    ax.set_title(f"Markkanen cloud, RoboPol: q/u vs distance (N={len(df)})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda mu_: 10 ** ((mu_ + 5) / 5) / 1000,
                   lambda d_kpc: 5 * np.log10(d_kpc * 1000) - 5)
    )
    sec.set_xlabel("Distance  [kpc]")

    plt.tight_layout()
    #plt.savefig(OUT_FIG, dpi=FIG_DPI, bbox_inches="tight")
    
    if outfile is not None:
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    else:
        plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
        
    plt.close(fig)


def plot_diagnostic(df: pd.DataFrame) -> float | None:
    dfd = cumulative_mahalanobis(df)
    b = simple_piecewise_linear_break(dfd["cum_dMaha"].to_numpy(),
                                      min_size=MIN_STARS_PER_SIDE)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(dfd)), dfd["cum_dMaha"], marker="o", ms=3, lw=1)
    if b is not None:
        d_b = float(dfd.loc[b, "r_med_photogeo"])
        ax.axvline(b, color="purple", lw=1.3)
        ax.text(b, ax.get_ylim()[1], f" {d_b:.0f} pc", color="purple",
                va="top", ha="left")
    ax.set_xlabel("Distance-sorted stellar index")
    ax.set_ylabel("Cumulative Mahalanobis distance")
    ax.set_title("Diagnostic only: not full TRIShUL breakpoint inference")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIAG, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return None if b is None else float(dfd.loc[b, "r_med_photogeo"])


def print_summary(df: pd.DataFrame, tests: pd.DataFrame, best_pc: float | None,
                  maha_pc: float | None) -> None:
    print("\nFinal sample")
    print("------------")
    print(f"N = {len(df)}")
    print(f"distance range = {df['r_med_photogeo'].min():.0f} - "
          f"{df['r_med_photogeo'].max():.0f} pc")
    print(f"median distance = {df['r_med_photogeo'].median():.0f} pc")

    manual = tests[tests["is_manual"]].copy()
    cols = ["cloud_pc", "n_before", "n_after", "q_before", "u_before",
            "q_after", "u_after", "delta_q", "delta_u", "jump_sig_diag"]
    print("\nManual cloud-distance tests")
    print("---------------------------")
    print(manual[cols].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    if best_pc is not None:
        best = tests[np.isclose(tests["cloud_pc"], best_pc)].iloc[0]
        print("\nAutomatic q/u step test")
        print("-----------------------")
        print(f"best split by piecewise weighted chi2: {best_pc:.0f} pc "
              f"(mu={best['cloud_mu']:.2f})")
        print(f"before/after counts: {int(best['n_before'])}/{int(best['n_after'])}")
        print(f"delta q = {best['delta_q']:.3f} ± {best['e_delta_q']:.3f} % "
              f"({best['sig_q']:.1f} sigma)")
        print(f"delta u = {best['delta_u']:.3f} ± {best['e_delta_u']:.3f} % "
              f"({best['sig_u']:.1f} sigma)")
        print(f"combined diagonal jump significance ≈ {best['jump_sig_diag']:.1f} sigma")
    else:
        print("\nAutomatic q/u step test: no valid split found.")

    if maha_pc is not None:
        print("\nCumulative Mahalanobis diagnostic")
        print("---------------------------------")
        print(f"simple piecewise-linear bend near {maha_pc:.0f} pc")
        print("This is only a quick diagnostic, not the full TRIShUL significance test.")



def main() -> None:
    df = load_and_prepare()
    if len(df) < 2 * MIN_STARS_PER_SIDE:
        raise ValueError("Too few stars after cuts for before/after tests.")

    tests = run_cloud_tests(df)
    best_pc = choose_best_cloud(tests)
    maha_pc = plot_diagnostic(df)

    df.to_csv(OUT_PREPARED, index=False)
    tests.to_csv(OUT_TESTS, index=False)
    #plot_main(df, tests, best_pc)
    #plot_main(df, tests, 430.0)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for d_pc in MANUAL_CLOUD_DISTANCES_PC:
        out_fig = os.path.join(OUT_DIR, f"markkanen_fig8_{int(d_pc)}pc.png")
        plot_main(df,tests, d_pc, outfile=out_fig)
    

    print_summary(df, tests, best_pc, maha_pc)
    print(f"\nSaved prepared table: {OUT_PREPARED}")
    print(f"Saved cloud tests:    {OUT_TESTS}")
    print(f"Saved main figure:    {OUT_FIG}")
    print(f"Saved diagnostic:     {OUT_DIAG}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

"""
plot_serkowski_fit.py

Plot the fitted Serkowski curves from the Fortran fitter on top of the
observed multiband polarization points. Parses out1.txt (the summary) for
the fit parameters, and reads the band data files for the points.

Run this AFTER ./serk has produced out1.txt.

Usage:
    python plot_serkowski_fit.py

Edit BAND_FILES, LAMBDA_EFF, TARGET_STARS, and the K coefficients (aK, bK)
to match what you used in code_fixed.for.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# Configuration

BAND_FILES = {
    "B": "../0_data/B/Mark_B_corr.dat",
    "G": "../0_data/G/Mark_g_corr.dat",   # G filter — confirm λ_eff 
    "R": "../0_data/R/Mark_R_corr.dat",
    "I": "../0_data/I/Mark_i_corr.dat",
}

    #"B": "../0_data/B/Markkanen_final.dat",
   # "G": "../0_data/G/Markkanen_final.dat",
   # "R": "../0_data/R/Markkanen_final_merged_serkowski.dat",
   # "I": "../0_data/I/Markkanen_final.dat",
LAMBDA_EFF = {
    "B": 0.436,
    "G": 0.477,
    "R": 0.640,
    "I": 0.763,
}

TARGET_STARS = ["Mark_26", "Mark_76", "Mark_81"]

# Must match aK, bK in code_fixed.for
aK = 1.86
bK = -0.10

OUT1_FILE = "out1.txt"
OUTPUT_PNG = f"serkowski_fit_aK{aK}_bK{bK}.png"


def serkowski(lam, Pmax, lam_max, aK, bK):
    K = aK * lam_max + bK
    return Pmax * np.exp(-K * np.log(lam_max / lam)**2)


def read_dat(filepath):
    records = defaultdict(list)
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split()
                if len(cols) < 4:
                    continue
                name = cols[0]
                try:
                    P = float(cols[2])
                    sP = float(cols[3])
                except ValueError:
                    continue
                if sP > 0:
                    records[name].append((P, sP))
    except FileNotFoundError:
        print(f"  File not found: {filepath}")
        return {}

    result = {}
    for name, measurements in records.items():
        Ps = np.array([m[0] for m in measurements])
        sPs = np.array([m[1] for m in measurements])
        w = 1.0 / sPs**2
        P_mean = np.sum(w * Ps) / np.sum(w)
        sP_mean = 1.0 / np.sqrt(np.sum(w))
        result[name] = (P_mean, sP_mean)
    return result


def parse_out1_last_block(filepath, target_stars):
    """
    out1.txt accumulates blocks from every run (status='append').
    For each target star, return the LAST entry found in the file.
    Returns dict: star -> (lam_max, sLam, Pmax, sP, chi2)
    """
    fits = {}
    pattern = re.compile(
        r"^\s*(\S+)\s+([\d.]+)\+-([\d.]+)\s+([\d.]+)\+-([\d.]+)\s+([\d.]+)\s*$"
    )
    with open(filepath) as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            name = m.group(1)
            if name in target_stars:
                fits[name] = (
                    float(m.group(2)),  # lam_max
                    float(m.group(3)),  # sLam
                    float(m.group(4)),  # Pmax
                    float(m.group(5)),  # sP
                    float(m.group(6)),  # chi2/dof
                )
    return fits


# Main

# Load band data
band_data = {}
for band, fpath in BAND_FILES.items():
    band_data[band] = read_dat(fpath)

bands_sorted = sorted(LAMBDA_EFF.keys(), key=lambda b: LAMBDA_EFF[b])

# Parse the latest fit results from out1.txt
fits = parse_out1_last_block(OUT1_FILE, TARGET_STARS)
print(f"Parsed fits from {OUT1_FILE}:")
for star, vals in fits.items():
    print(f"  {star}: lam_max={vals[0]:.3f}+-{vals[1]:.3f}, "
          f"Pmax={vals[2]:.3f}+-{vals[3]:.3f}, chi2/dof={vals[4]:.2f}")

# Plot
lam_fine = np.linspace(0.3, 1.0, 300)

fig, axes = plt.subplots(1, len(TARGET_STARS), figsize=(5 * len(TARGET_STARS), 4),
                         sharey=False)
if len(TARGET_STARS) == 1:
    axes = [axes]

for ax, star in zip(axes, TARGET_STARS):
    # Data points
    lams, Ps, sPs = [], [], []
    for band in bands_sorted:
        if band not in band_data or star not in band_data[band]:
            continue
        lams.append(LAMBDA_EFF[band])
        P, sP = band_data[band][star]
        Ps.append(P)
        sPs.append(sP)
        ax.annotate(band, (LAMBDA_EFF[band], P),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=9, color="C0")

    if len(Ps) == 0:
        ax.set_title(f"{star}\n(no data)")
        continue

    ax.errorbar(lams, Ps, yerr=sPs, fmt="o", color="C0", capsize=4,
                label="Observed P", zorder=3)

    # Fitted curve
    if star in fits:
        lam_max, sLam, Pmax, sP, chi2 = fits[star]
        P_fit = serkowski(lam_fine, Pmax, lam_max, aK, bK)
        K_eff = aK * lam_max + bK
        ax.plot(lam_fine, P_fit, "r-", lw=1.8,
                label=(f"Serkowski fit\n"
                       f"$\\lambda_{{max}}$={lam_max:.3f}$\\pm${sLam:.3f} $\\mu$m\n"
                       f"$P_{{max}}$={Pmax:.2f}$\\pm${sP:.2f}%\n"
                       f"K={K_eff:.2f}, $\\chi^2$/dof={chi2:.2f}"))
        # Mark the peak location
        ax.axvline(lam_max, color="r", linestyle=":", alpha=0.4)
    else:
        ax.text(0.5, 0.95, "no fit found in out1.txt",
                transform=ax.transAxes, ha="center", va="top", color="red")

    ax.set_xlabel("$\\lambda$ ($\\mu$m)")
    ax.set_ylabel("P (%)")
    ax.set_title(star)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlim(0.35, 0.95)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

plt.suptitle(f"Serkowski fit (K = {aK}$\\lambda_{{max}}$ + {bK})", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved {OUTPUT_PNG}")
plt.show()

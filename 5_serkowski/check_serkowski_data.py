"""
check_serkowski_data.py

Quick diagnostic: plot P(lambda) for each star from your multiband .dat files
BEFORE running the Fortran fitter. This lets you see whether the data actually
looks Serkowski-shaped (peaks in V/R band) or has problems.

Usage:
    python check_serkowski_data.py

Edit BAND_FILES and TARGET_STARS below to match your setup.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration 

BAND_FILES = {
    "B": "../0_data/B/Mark_B_corr.dat",
    "G": "../0_data/G/Mark_g_corr.dat",   # G filter — confirm λ_eff 
    "R": "../0_data/R/Mark_R_corr.dat",
    "I": "../0_data/I/Mark_i_corr.dat",
}


LAMBDA_EFF = {
    "B": 0.436,     #B - Johnson
    "G": 0.477,     #g - sdss-g' / Sloan  g
    "R": 0.640,     #R - Cousins
    "I": 0.763,     #i - sdss-i' / Sloan i
}

TARGET_STARS = ["Mark_26", "Mark_76", "Mark_81"]

# Serkowski curve for reference (typical ISM: Pmax=1%, lam_max=0.55 um)
def serkowski(lam, Pmax, lam_max, aK=1.86, bK=-0.10):
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
                    P  = float(cols[2])
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
        Ps  = np.array([m[0] for m in measurements])
        sPs = np.array([m[1] for m in measurements])
        w   = 1.0 / sPs**2
        P_mean  = np.sum(w * Ps) / np.sum(w)
        sP_mean = 1.0 / np.sqrt(np.sum(w))
        result[name] = (P_mean, sP_mean)
    return result


band_data = {}
for band, fpath in BAND_FILES.items():
    band_data[band] = read_dat(fpath)

bands_sorted = sorted(LAMBDA_EFF.keys(), key=lambda b: LAMBDA_EFF[b])
lam_fine = np.linspace(0.3, 1.0, 200)

fig, axes = plt.subplots(1, len(TARGET_STARS), figsize=(5 * len(TARGET_STARS), 4),
                         sharey=False)
if len(TARGET_STARS) == 1:
    axes = [axes]

for ax, star in zip(axes, TARGET_STARS):
    lams, Ps, sPs = [], [], []
    for band in bands_sorted:
        if band not in band_data or star not in band_data[band]:
            continue
        lams.append(LAMBDA_EFF[band])
        P, sP = band_data[band][star]
        Ps.append(P)
        sPs.append(sP)
        ax.annotate(band, (LAMBDA_EFF[band], P),
                    textcoords="offset points", xytext=(4, 4), fontsize=9, color="C0")

    if len(Ps) == 0:
        ax.set_title(f"{star}\n(no data found)")
        continue

    ax.errorbar(lams, Ps, yerr=sPs, fmt="o", color="C0", capsize=4,
                label="Observed P")

    # Overplot a reference Serkowski curve with Pmax = max(P), lam_max = lam at max P
    idx_max = np.argmax(Ps)
    P_ref   = Ps[idx_max]
    lam_ref = lams[idx_max]
    P_serk  = serkowski(lam_fine, P_ref, lam_ref)
    ax.plot(lam_fine, P_serk, "r--", alpha=0.6,
            label=f"Serkowski ref\n(λ_max={lam_ref:.2f}, Pmax={P_ref:.2f}%)")

    ax.axvline(0.55, color="gray", linestyle=":", alpha=0.5, label="Typical λ_max=0.55")
    ax.set_xlabel("λ (μm)")
    ax.set_ylabel("P (%)")
    ax.set_title(star)
    ax.legend(fontsize=8)
    ax.set_xlim(0.35, 0.90)

    # Print a warning if the curve is not peaked in the middle
    if lam_ref == max(lams):
        print(f"WARNING {star}: P is highest at the reddest band ({bands_sorted[-1]}). "
              f"The data may not be instrument-corrected, or λ_max > I-band.")
    if lam_ref == min(lams):
        print(f"WARNING {star}: P is highest at the bluest band ({bands_sorted[0]}). "
              f"Unusual — check data.")

plt.suptitle("P(λ) diagnostic — check shape before Serkowski fit", fontsize=11)
plt.tight_layout()
plt.savefig("serkowski_diagnostic.png", dpi=150, bbox_inches="tight")
print("Saved serkowski_diagnostic.png")
plt.show()

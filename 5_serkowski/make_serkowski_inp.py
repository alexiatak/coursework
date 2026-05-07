"""
make_serkowski_inp.py

Convert multiband RoboPol Markkanen_final.dat files to the inp.txt format
required by the Fortran Serkowski fitting program SERK_OBS_2.

Usage:
    python make_serkowski_inp.py

Edit the BAND_FILES and LAMBDA_EFF dictionaries at the top to match your
actual file paths and filter wavelengths.

Output: inp.txt (ready to feed to the Fortran program)
        out1.txt and out2.txt (empty files required by Fortran)
"""

import numpy as np
from collections import defaultdict

#  Configuration 

# Map each band name to its final (instrument-corrected, merged) .dat file.
# Adjust paths to match your actual file locations.
BAND_FILES = {
    "B": "../0_data/B/Markkanen_final.dat",
    "G": "../0_data/G/Markkanen_final.dat",   # G filter — confirm λ_eff 
    "R": "../0_data/R/Markkanen_final_merged_serkowski.dat",
    "I": "../0_data/I/Markkanen_final.dat",
}

# Effective wavelengths in microns.
# B and I are standard Cousins/Johnson. 
# G (green) — check if it is Cousins V, use 0.530; 
#             
LAMBDA_EFF = {
    "B": 0.440,
    "G": 0.530,   # confirm this 
    "R": 0.640,
    "I": 0.800,
}

# Which stars to fit. These must appear by name in the .dat files.
TARGET_STARS = ["Mark_26", "Mark_76", "Mark_81"]

# Output file
OUT_INP = "inp.txt"

#  Reading 

def read_dat(filepath):
    """
    Read a Markkanen_final.dat file (or merged equivalent).
    Returns a dict: star_name -> (P_percent, sP_percent)
    If a star appears multiple times (not yet merged), computes weighted mean.
    """
    records = defaultdict(list)
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
                P  = float(cols[2])   # P[%]
                sP = float(cols[3])   # sP[%]
            except ValueError:
                continue
            if sP <= 0:
                print(f"  Warning: zero or negative sP for {name} in {filepath}, skipping.")
                continue
            records[name].append((P, sP))

    result = {}
    for name, measurements in records.items():
        if len(measurements) == 1:
            result[name] = measurements[0]
        else:
            # Inverse-variance weighted mean
            Ps  = np.array([m[0] for m in measurements])
            sPs = np.array([m[1] for m in measurements])
            w   = 1.0 / sPs**2
            P_mean  = np.sum(w * Ps) / np.sum(w)
            sP_mean = 1.0 / np.sqrt(np.sum(w))
            result[name] = (P_mean, sP_mean)
            print(f"  {name}: {len(measurements)} measurements merged -> "
                  f"P={P_mean:.3f} +/- {sP_mean:.3f} %")
    return result



def main():
    # Load all bands
    band_data = {}
    for band, filepath in BAND_FILES.items():
        print(f"Reading {band} band from {filepath} ...")
        try:
            band_data[band] = read_dat(filepath)
        except FileNotFoundError:
            print(f"  WARNING: file not found, skipping band {band}.")
            band_data[band] = {}

    # Determine band order by wavelength (ascending)
    bands_sorted = sorted(LAMBDA_EFF.keys(), key=lambda b: LAMBDA_EFF[b])

    # Build inp.txt
    print(f"\nWriting {OUT_INP} ...")
    lines = []
    for star in TARGET_STARS:
        # Collect available (lambda, P, sP) for this star
        points = []
        for band in bands_sorted:
            if band not in band_data:
                continue
            if star not in band_data[band]:
                print(f"  WARNING: {star} not found in band {band}, skipping that point.")
                continue
            lam = LAMBDA_EFF[band]
            P, sP = band_data[band][star]
            points.append((lam, P, sP))

        n = len(points)
        if n < 3:
            print(f"  WARNING: {star} has only {n} band(s) — Serkowski fit needs >= 3. Skipping.")
            continue

        # Fortran format: star name in 8 chars (truncate if needed), n as I4
        name_8 = star[:8].ljust(8)        # A8 field
        header = f" {name_8}{n:4d}"
        lines.append(header)

        for lam, P, sP in points:
            # Format: (1x, 2f6.3, f7.3)
            # 1x blank, then lambda as f6.3 (6 chars, 3 decimal), P as f6.3, sP as f7.3
            row = f" {lam:6.3f}{P:6.3f}{sP:7.3f}"
            lines.append(row)
            print(f"    {star}  {band}  λ={lam:.3f}  P={P:.3f}  sP={sP:.3f}")

    with open(OUT_INP, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Create empty output files (Fortran opens them with status='old')
    for fname in ["out1.txt", "out2.txt"]:
        open(fname, "a").close()
        print(f"Created/touched {fname}")

    print(f"\nDone. Now compile and run the Fortran program:")
    print(f"  gfortran code.for -o serk")
    print(f"  ./serk")
    print(f"\nResults will be in out1.txt (summary) and out2.txt (per-point fit).")


if __name__ == "__main__":
    main()

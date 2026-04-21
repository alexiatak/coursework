"""
smooth_planck_353.py

Downgrade and smooth the Planck PR3 353 GHz full-mission sky map (I, Q, U)
to a 20 arcmin Gaussian beam at Nside=1024, for comparison with optical
polarization measurements toward the Markkanen molecular cloud.

Based on a script shared by Raphael Skalidis. 

Input file to download manually from:
  https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/
  -> HFI_SkyMap_353_2048_R3.01_full.fits   (~2 GB)

Usage:
    python smooth_planck_353.py
"""

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


# ------------------------- Parameters to change -----------------------------

DATA_DIR     = os.path.expanduser("~/Desktop/extra_data")
SMOOTH_DIR     = os.path.expanduser("~/Desktop/coursework/3_planck")
PLANCK_FILE  = os.path.join(DATA_DIR, "HFI_SkyMap_353_2048_R3.01_full.fits")

# Where smoothed outputs will be written 
OUT_DIR      = SMOOTH_DIR

TARGET_NSIDE = 1024          # downgrade from native 2048
TARGET_RES   = 20.0 / 60.0   # target FWHM in degrees (= 20 arcmin)
PLANCK_BEAM  = 5.0  / 60.0   # native 353 GHz FWHM in degrees (~4.82', rounded)

# Markkanen field center, for the sanity-check plot
MK_L_DEG = 265.0
MK_B_DEG = 80.0
MK_XSIZE = 150          #pixels per side in gnomview
MK_RESO = 11.0          #arcmin per pixel (~27.5 deg field)
# ----------------------------------------------------------------------------


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(PLANCK_FILE):
        raise FileNotFoundError(
            f"Planck map not found at {PLANCK_FILE}\n"
            "Download HFI_SkyMap_353_2048_R3.01_full.fits from\n"
            "  https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/\n"
            "and either place it at that path or edit DATA_DIR / PLANCK_FILE at the "
            "top of this script."
        )

    # ---- 1. Read I, Q, U ---------------------------------------------------
    # Polarized Planck HFI maps have 10 fields; 0,1,2 are I_Stokes, Q_Stokes, U_Stokes.
    # At 353 GHz all three are in K_CMB (not MJy/sr).
    print("Reading I, Q, U from", PLANCK_FILE)
    I_in, Q_in, U_in = hp.read_map(PLANCK_FILE, field=(0, 1, 2))
    nside_in = hp.get_nside(I_in)
    print(f"  native nside = {nside_in}, npix = {hp.nside2npix(nside_in)}")

    # ---- 2. Downgrade to target_nside (in NESTED for ud_grade efficiency) --
    # The input file is stored in NESTED ordering (Planck convention), but
    # hp.read_map returns it reordered to RING by default. Work in RING
    # throughout and later save the data in NESTED.
    print(f"Downgrading to nside = {TARGET_NSIDE} ...")
    I_dg = hp.ud_grade(I_in, nside_out=TARGET_NSIDE, order_in='RING', order_out='RING')
    Q_dg = hp.ud_grade(Q_in, nside_out=TARGET_NSIDE, order_in='RING', order_out='RING')
    U_dg = hp.ud_grade(U_in, nside_out=TARGET_NSIDE, order_in='RING', order_out='RING')

    # ---- 3. Smooth with a Gaussian kernel ----------------------------------
    # We want the final effective beam to be TARGET_RES. Since the map already has a native beam of ~PLANCK_BEAM, we convolve with a Gaussian of
    # FWHM = sqrt(target^2 - native^2) so that the beams add in quadrature.
    smooth_fwhm_deg = np.sqrt(TARGET_RES**2 - PLANCK_BEAM**2)
    print(f"Convolving with Gaussian FWHM = {smooth_fwhm_deg*60:.3f} arcmin "
          f"(target {TARGET_RES*60:.1f}', native {PLANCK_BEAM*60:.1f}')")

    # pol=True : it tells healpy to smooth Q,U as a spin-2 field,
    # Do NOT pass nest=True here
    # hp.smoothing works on RING ordering only.
    I_sm, Q_sm, U_sm = hp.smoothing(
        [I_dg, Q_dg, U_dg],
        fwhm=np.radians(smooth_fwhm_deg),
        pol=True,
    )

    # ---- 4. Write outputs --------------------------------------------------
    # Match Planck convention of writing in NESTED ordering
    
    I_sm_n = hp.reorder(I_sm, r2n=True)
    Q_sm_n = hp.reorder(Q_sm, r2n=True)
    U_sm_n = hp.reorder(U_sm, r2n=True)

    tag = f"smoothed{int(TARGET_RES*60):02d}arcmin_nside{TARGET_NSIDE}_nested"
    out_I = os.path.join(OUT_DIR, f"planck_353_I_{tag}.fits")
    out_Q = os.path.join(OUT_DIR, f"planck_353_Q_{tag}.fits")
    out_U = os.path.join(OUT_DIR, f"planck_353_U_{tag}.fits")

    hp.write_map(out_I, I_sm_n, nest=True, coord='G',
                 column_units='K_CMB', overwrite=True)
    hp.write_map(out_Q, Q_sm_n, nest=True, coord='G',
                 column_units='K_CMB', overwrite=True)
    hp.write_map(out_U, U_sm_n, nest=True, coord='G',
                 column_units='K_CMB', overwrite=True)
    print("Wrote:")
    for f in (out_I, out_Q, out_U):
        print(f"  {f}  ({os.path.getsize(f)/1e6:.1f} MB)")

    # ---- 5. Sanity-check plot ----------------------------------------------
    # gnomview of the Markkanen field, reprojected to galactic to match
    # our other plots.
    print("Making sanity-check plot ...")
    fig = plt.figure(figsize=(10, 5))
    hp.gnomview(
        I_sm,                       # RING ordering, as gnomview expects by default
        rot=(MK_L_DEG, MK_B_DEG),   # center on Markkanen
        xsize=MK_XSIZE, ysize=MK_XSIZE,
        reso=MK_RESO,  # arcmin per pixel
        coord='G',          
        title=f"Planck 353 GHz I, smoothed to {TARGET_RES*60:.0f}' "
              f"(Markkanen field, galactic)",
        unit='K_CMB',
        cmap='inferno',
        notext=False,
        sub=(1, 2, 1),
    )
    hp.graticule(dpar=2, dmer=2, color='white', alpha=0.3)

    # show polarized intensity p_I = sqrt(Q^2 + U^2) as a quick sanity check
    P_sm = np.sqrt(Q_sm**2 + U_sm**2)
    hp.gnomview(
        P_sm,
        rot=(MK_L_DEG, MK_B_DEG),
        xsize=MK_XSIZE, ysize=MK_XSIZE,
        reso=MK_RESO,
        coord='G',
        title="Planck 353 GHz polarized intensity sqrt(Q^2+U^2)",
        unit='K_CMB',
        cmap='viridis',
        notext=False,
        sub=(1, 2, 2),
    )
    hp.graticule(dpar=2, dmer=2, color='white', alpha=0.3)

    out_plot = os.path.join(OUT_DIR, f"planck_353_markkanen_sanity_{tag}.png")
    plt.savefig(out_plot, dpi=120, bbox_inches='tight')
    print(f"  {out_plot}")

    print("\nDone.")
    


if __name__ == "__main__":
    main()

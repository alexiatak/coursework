# Planck 353 GHz smoothing

Script to produce smoothed Planck PR3 353 GHz I, Q, U maps at 20' resolution,
for comparison with optical polarization measurements toward the Markkanen
cloud. Approach follows a script shared by Raphael Skalidis, extended to the
full (I, Q, U) triple so we can compare both polarization fraction and EVPA.

## 1. Download the Planck map

The raw Planck full-mission map is ~2 GB. 
Keep it in a separate local data directory and point the script at it.

1. Go to
   https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/
2. Download `HFI_SkyMap_353_2048_R3.01_full.fits`.
3. Place it outside your repo, e.g. `~/Desktop/extra_data/`.

The 353 GHz full-channel map is in HEALPix format, Nside=2048, Galactic
coordinates, NESTED ordering. Fields 0, 1, 2 are I, Q, U in K_CMB. The native
beam FWHM is ~4.82' (we approximate as 5').

## 2. Run the smoothing

```
conda activate mark 
python smooth_planck_353.py
```

Edit the parameter block at the top of the script if your data lives
somewhere other than `~/Desktop/extra_data/`. Output is written next to the input
file: three FITS maps (I, Q, U smoothed to 20' at Nside=1024, NESTED) plus a
PNG sanity-check plot of the Markkanen field.

## 3. What the script actually does

- Reads the three Stokes parameters together.
- Downgrades from Nside=2048 to 1024.
- Smooths with a Gaussian kernel of FWHM = sqrt(20'^2 - 5'^2) so that the
  effective output beam is 20'.
- Uses `pol=True` in `hp.smoothing`, which treats Q and U as a spin-2 field
  (correct handling of linear polarization under rotation — not the same as
  independently smoothing them as scalars).
- Writes each map to its own file. **Units are K_CMB** (MJy/sr only applies to the 545 and
  857 GHz channels).



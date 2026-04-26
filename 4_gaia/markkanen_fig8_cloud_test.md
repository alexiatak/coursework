# markkanen_fig8_cloud_test.py

Fig. 8-style diagnostic plot for the Markkanen cloud, in the spirit of Uppal+2025 (TRIShUL- https://arxiv.org/pdf/2510.25911). 
This is not the full TRIShUL pipeline. It produces the q/u vs distance modulus plot and runs a few cloud-distance tests around it.

## Inputs

- `merged_output.csv` — RoboPol table with Name, q, sq, u, su, ra, dec
- `bj_distances.csv` — Bailer-Jones distances with Name, gid, r_med_photogeo, r_lo_photogeo, r_hi_photogeo

Paths are set at the top of the script in the CONFIG block.

## What it does

1. Reads both tables and joins them on `Name`.
2. Converts q, u, sq, su from fractional to percent if they look fractional.
3. Computes distance modulus `mu` and its uncertainty from r_lo and r_hi.
4. Optional quality cut on relative distance error (default 25%).
5. Sorts by distance.
6. For each manual cloud distance in `MANUAL_CLOUD_DISTANCES_PC`, splits the sample at that distance and computes weighted means of q and u before and after, the jump, and a piecewise chi-squared.
7. Picks the cloud distance with the smallest piecewise chi-squared as the "best" automatic split.
8. Computes a cumulative Mahalanobis-like quantity over distance-sorted index and finds a simple piecewise-linear bend as a separate diagnostic.
9. Saves a Fig. 8 plot for every manual distance and one diagnostic plot.

## Outputs

- `markkanen_fig8_prepared.csv` — joined table with mu, sigma_d, etc.
- `markkanen_cloud_distance_tests.csv` — one row per tested cloud distance with all the split statistics.
- `markkanen_fig8_distance_tests/markkanen_fig8_<d>pc.png` — one plot per tested distance.
- `markkanen_cumulative_mahalanobis.png` — cumulative Mahalanobis diagnostic.

## Plot

q (green) and u (blue) vs distance modulus, with errorbars. A secondary x-axis shows distance in kpc. The tested cloud distance is drawn as a purple vertical line; weighted means before and after are drawn as horizontal segments at the q and u levels.


## Notes

 
- `MIN_STARS_PER_SIDE` is 8 by default. Splits with fewer stars on either side are skipped.
- The cumulative Mahalanobis diagnostic uses `sqrt((q/sq)^2 + (u/su)^2)` summed over index, not the full covariance form. Good enough as a quick visual check.

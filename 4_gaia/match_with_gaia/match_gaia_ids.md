# match_gaia_ids.py

Crossmatches the 154 RoboPol observations against Dima's observing list to attach a Gaia source ID to each star.

## Why

`merged_output.csv` only has Name, RA, Dec and the polarization columns. To get Bailer-Jones distances we need Gaia source IDs.

## Inputs

- `merged_output.csv` — RoboPol observations (Name, P, PA, q, u, ra, dec, ...)
- `Observing_list_small_correct_gid.txt` — Dima's list, whitespace-separated, columns: gaia_id, ra, dec, l, b, rp

## What it does

1. Loads both tables.
2. Builds SkyCoord objects for both.
3. For each observed star, finds the nearest entry in Dima's list using `match_coordinates_sky`.
4. Keeps the match if separation is below `MAX_SEP_ARCSEC` (default 2 arcsec).
5. Writes the result with a `gaia_id` column and a `matched` boolean.

The Gaia ID is read as a string the whole way through to avoid the 19-digit int losing precision.

## Output

`merged_with_gaiaid.csv` — original columns plus `gaia_id`, `sep_arcsec`, `matched`.

## Notes

- Unmatched stars (typically a handful) are kept in the output with `matched=False` and an empty `gaia_id`. They get filtered out in the next step.
- Run this before `prepare_for_gaia_archive.py`.

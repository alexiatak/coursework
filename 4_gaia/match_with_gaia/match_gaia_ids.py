#!/usr/bin/env python3
"""

Crossmatch 154 RoboPol observations (merged_output.csv) against Dima's
observing list (Observing_list_small_correct_gid.txt, 299 targets w/ Gaia IDs)
to assign a Gaia DR3 source_id to each observed star.

What the script produces:
1.  merged_with_gaiaid.csv
        All rows from merged_output.csv, plus columns
            gaia_id, l, b, rp_obslist, match_sep_arcsec
2.  gaia_id_list.csv
        One-column file with just the gaia_id values — this is the file
        you upload to the Gaia archive (https://gea.esac.esa.int/archive/)
        or to TOPCAT to retrieve rpgeo, b_rpgeo, B_rpgeo.

"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord

# =============================================================================
# EDIT THESE PATHS
# =============================================================================
MERGED_CSV  = "../../2_sky_plot/merged_output.csv"
OBS_LIST    = "Observing_list_small_correct_gid.txt"
OUT_MATCHED = "merged_with_gaiaid.csv"
OUT_IDLIST  = "gaia_id_list.csv"
OUT_UNMATCHED = "unmatched_for_gaia_cone_search.csv"

# Crossmatch radius: tighten to 1" if you
# want to be strict, loosen only if you see lots of no match warnings.
MAX_SEP_ARCSEC = 2.0


def read_observing_list(path: str) -> pd.DataFrame:
    """
    Load Dima's observing list.

    Format (whitespace-separated, single-line header starting with '#'):
        # GID   RA    DEC     l     b     RP
        3954447376646676096 190.49023 20.28698 284.84923 82.82366 11.80
        ...
        
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Observing list not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["gaia_id", "ra", "dec", "l", "b", "rp"],
        dtype={"gaia_id": str},
        engine="python",
    )
    # Defensive: strip any stray whitespace from the Gaia ID strings
    df["gaia_id"] = df["gaia_id"].str.strip()
    return df


def read_merged(path: str) -> pd.DataFrame:
    """
    Load merged_output.csv. Expected columns:
        Name, P[%], PA[deg], q, u, ra, dec
    (plus any optional sigma columns)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"merged_output.csv not found: {path}")

    # Try automatic delimiter detection first (handles both CSV and TSV)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)

    # Some RoboPol pipelines produce 'RA','DEC' (uppercase). Normalize to ra/dec.
    if "ra" not in df.columns:
        for alt in ("RA", "Ra"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ra"})
                break
    if "dec" not in df.columns:
        for alt in ("DEC", "Dec"):
            if alt in df.columns:
                df = df.rename(columns={alt: "dec"})
                break

    required = ["Name", "ra", "dec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"merged_output.csv is missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    return df


def crossmatch(merged: pd.DataFrame, obs: pd.DataFrame,
               max_sep_arcsec: float) -> pd.DataFrame:
    """
    For each row in `merged`, find the nearest entry in `obs` on the sky.
    Return `merged` with added columns:
        gaia_id          — matched Gaia DR3 source_id (string, or empty)
        l, b             — Galactic coordinates from the observing list
        rp_obslist       — RP magnitude from the observing list
        match_sep_arcsec — separation of the match in arcsec
        matched          — True if separation < max_sep_arcsec
        has_coords       — True if the input row had non-NaN RA/Dec

    match_coordinates_sky: a KD-tree nearest-neighbor search on the unit sphere. 
    It returns, for each element of the first array,
    the index of the closest element in the second array and the separation.

    
    Rows with NaN RA or Dec  are kept in the output but marked as unmatched with
    has_coords=False. 
    """
    has_coords = merged["ra"].notna() & merged["dec"].notna()

    # Run the KD-tree match only on rows with valid coordinates.
    # We then write results back into a full-length array aligned with `merged`.
    out = merged.copy()
    out["gaia_id"]          = ""
    out["l"]                = np.nan
    out["b"]                = np.nan
    out["rp_obslist"]       = np.nan
    out["match_sep_arcsec"] = np.nan
    out["matched"]          = False
    out["has_coords"]       = has_coords.to_numpy()

    if has_coords.sum() == 0:
        return out

    valid = merged.loc[has_coords]
    c_valid = SkyCoord(ra=valid["ra"].to_numpy() * u.deg,
                       dec=valid["dec"].to_numpy() * u.deg,
                       frame="icrs")
    c_obs = SkyCoord(ra=obs["ra"].to_numpy() * u.deg,
                     dec=obs["dec"].to_numpy() * u.deg,
                     frame="icrs")

    idx, sep2d, _ = c_valid.match_to_catalog_sky(c_obs)
    sep_arcsec = sep2d.to(u.arcsec).value
    good = sep_arcsec < max_sep_arcsec

    # Scatter the per-valid-row results back into the full output frame,
    # using `has_coords` as the boolean mask over the full index.
    valid_idx = out.index[has_coords]
    out.loc[valid_idx, "match_sep_arcsec"] = sep_arcsec
    out.loc[valid_idx, "matched"]          = good
    out.loc[valid_idx[good], "gaia_id"]    = obs["gaia_id"].to_numpy()[idx[good]]
    out.loc[valid_idx[good], "l"]          = obs["l"].to_numpy()[idx[good]]
    out.loc[valid_idx[good], "b"]          = obs["b"].to_numpy()[idx[good]]
    out.loc[valid_idx[good], "rp_obslist"] = obs["rp"].to_numpy()[idx[good]]

    return out


def summarize(matched: pd.DataFrame, max_sep_arcsec: float) -> None:
    """Print a one-page diagnostic of the match quality."""
    n = len(matched)
    has_coords = matched["has_coords"]
    n_no_coords = int((~has_coords).sum())
    ok = int(matched["matched"].sum())
    # Unmatched-with-coords = had valid RA/Dec but nothing nearby in the obs list
    bad = int((has_coords & ~matched["matched"]).sum())

    print("\n" + "=" * 60)
    print("Crossmatch summary")
    print("=" * 60)
    print(f"Observations:                {n}")
    print(f"  Missing RA/Dec (skipped):  {n_no_coords}")
    print(f"  Matched (< {max_sep_arcsec}\"):          {ok}")
    print(f"  No match within {max_sep_arcsec}\":     {bad}")

    if ok > 0:
        good = matched.loc[matched["matched"]]
        sep = good["match_sep_arcsec"].to_numpy()
        print(f"\nMatch separations (arcsec):")
        print(f"  median   : {np.median(sep):.3f}")
        print(f"  mean     : {np.mean(sep):.3f}")
        print(f"  90th pct : {np.percentile(sep, 90):.3f}")
        print(f"  max      : {np.max(sep):.3f}")

        # Duplicate Gaia ID check: did two different observations land on the
        # same target in the observing list? If yes, that's either multi-epoch
        # (fine — multiple JDs for one star) or a genuine ambiguity worth
        # looking at. We report both cases.
        dup_counts = good["gaia_id"].value_counts()
        multi = dup_counts[dup_counts > 1]
        if len(multi) > 0:
            print(f"\n{len(multi)} Gaia IDs have multiple observations:")
            print(f"  (this is normal if you have multi-epoch data — "
                  f"check 'JD' in merged_output.csv)")
            for gid, cnt in multi.head(5).items():
                names = good.loc[good["gaia_id"] == gid, "Name"].tolist()
                print(f"  {gid}: {cnt} obs  Names: {names}")
            if len(multi) > 5:
                print(f"  ... and {len(multi) - 5} more")

    if n_no_coords > 0:
        print(f"\nRows with missing RA/Dec (not crossmatched):")
        no_coords = matched.loc[~has_coords, ["Name", "JD"]]
        for _, r in no_coords.iterrows():
            jd_str = f"{r['JD']:.4f}" if pd.notna(r.get('JD')) else "?"
            print(f"  {r['Name']:25s}  JD={jd_str}")
        print(f"  (these are typically multi-epoch '_merged' rows whose")
        print(f"   coordinates weren't carried over from the upstream pipeline)")

    if bad > 0:
        print(f"\nRows with coords but no match within {max_sep_arcsec}\":")
        unmatched = matched.loc[has_coords & ~matched["matched"],
                                ["Name", "ra", "dec", "match_sep_arcsec"]]
        for _, r in unmatched.head(50).iterrows():
            print(f"  {r['Name']:25s}  RA={r['ra']:.5f}  Dec={r['dec']:.5f}  "
                  f"nearest neighbour at {r['match_sep_arcsec']:.2f}\"")
        if len(unmatched) > 15:
            print(f"  ... and {len(unmatched) - 15} more")
        print("\nIf many rows are unmatched, possible causes:")
        print("  * Observing list is incomplete for your observed targets")


    print("=" * 60)


def main() -> None:
    print(f"Reading merged observations from: {MERGED_CSV}")
    merged = read_merged(MERGED_CSV)
    print(f"  -> {len(merged)} rows, columns: {list(merged.columns)}")

    print(f"\nReading observing list from:      {OBS_LIST}")
    obs = read_observing_list(OBS_LIST)
    print(f"  -> {len(obs)} targets")

    print(f"\nCrossmatching with tolerance {MAX_SEP_ARCSEC}\"...")
    matched = crossmatch(merged, obs, MAX_SEP_ARCSEC)

    summarize(matched, MAX_SEP_ARCSEC)

    # Write the full matched table
    matched.to_csv(OUT_MATCHED, index=False)
    print(f"\nSaved matched table -> {OUT_MATCHED}")

    # Write the bare Gaia ID list for uploading to the Gaia archive / TOPCAT.
    # Only include successfully matched rows; deduplicate so multi-epoch
    # observations of the same star contribute a single ID.
    ok_ids = (matched.loc[matched["matched"], "gaia_id"]
                     .drop_duplicates()
                     .reset_index(drop=True))
    # Save with header 'source_id' — this matches the column name used in
    # external.gaiaedr3_distance, so your ADQL JOIN "ON t.source_id = bj.source_id"
    # just works without a column rename.
    ok_ids.to_frame(name="source_id").to_csv(OUT_IDLIST, index=False)
    print(f"Saved Gaia ID list  -> {OUT_IDLIST}  ({len(ok_ids)} unique IDs)")

    # Write the unmatched-but-has-coords rows to a separate CSV. These are
    # stars with valid RA/Dec that aren't in the observing list — candidates
    # for a fallback astroquery cone search against Gaia itself.
    unmatched = matched.loc[matched["has_coords"] & ~matched["matched"],
                            ["Name", "ra", "dec", "JD"]]
    if len(unmatched) > 0:
        unmatched.to_csv(OUT_UNMATCHED, index=False)
        print(f"Saved unmatched     -> {OUT_UNMATCHED}  ({len(unmatched)} stars)")

    print("\nNext step:")
    print(f"  1. run patch_dima_gaia_ids.py with new coordinates")
    print(f"  2. run prepare_for_gaia_archive.py")
    

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

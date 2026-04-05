#!/usr/bin/env python3
"""
Agglomeration polygon selector with crossmatch.

Pipeline
--------
1. Read the Panopoulou et al. 2025 optical polarization catalog
   (Unique_Source_Pol_Distance_Table.csv)
2. Convert RA/Dec -> Galactic coordinates (l, b)
3. Keep only stars inside a predefined sky polygon
4. Save the selected external stars to a CSV
5. Read my own observations from merged_output.csv
6. Crossmatch: for each of my stars find the nearest Panopoulou match
7. Save the matched pairs to a second CSV
8. Optionally make a diagnostic plot

Catalogs used

INPUT 1 — Panopoulou et al. 2025  (Unique_Source_Pol_Distance_Table.csv)
INPUT 2 — merged_output.csv  (your own observations)
   
OUTPUT 1 — external_panopoulou_<polygon>.csv (All Panopoulou stars inside the polygon)
OUTPUT 2 — crossmatched_<polygon>.csv (One row per observed star that has a Panopoulou counterpart, if there is one)

Dependencies
    pip install astropy pandas numpy matplotlib pygplates
    pip install healpy   # optional — only for the FITS background plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pygplates
from astropy import units as u
from astropy.coordinates import SkyCoord

try:
    import healpy as hp
except ImportError:
    hp = None



# merged_output.csv column names — edit these if your headers differ
MERGED_NAME_COL = "Name"
MERGED_RA_COL   = "ra"
MERGED_DEC_COL  = "dec"


# Sky polygon presets  — (Galactic longitude l, Galactic latitude b) degrees
ORIGINAL_POLYGON: List[Tuple[float, float]] = [
    (357.2, 81.5),
    (335.0, 89.1),
    (228.0, 87.0),
    (246.0, 80.0),
    (237.0, 66.0),
    (265.0, 67.0),
    (275.0, 79.5),
    (296.5, 71.8),
    (326.7, 82.6),
    (350.0, 77.8),
]

EXPANDED_POLYGON: List[Tuple[float, float]] = [
    (359.0, 82.0),
    (333.0, 89.3),
    (224.0, 87.5),
    (243.0, 79.0),
    (234.0, 64.5),
    (268.0, 65.5),
    (280.0, 76.5),      
    (316.0, 79.0),      
    (335.0, 81.5),
    (353.0, 78.5),
]


# Panopoulou catalog — required columns (exact header spelling)
PANOPOULOU_REQUIRED = [
    "EDR3_source_id",   
    "starID",
    "RA",
    "Dec",
    "p",
    "e_p",
    "evpa",
    "e_evpa",
]

PANOPOULOU_DIST_COLS = [
    "r_med_geo", "r_lo_geo", "r_hi_geo",
    "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo",
]


@dataclass
class SelectionResult:
    selected: pd.DataFrame
    polygon: List[Tuple[float, float]]



def load_panopoulou_catalog(path: Path) -> pd.DataFrame:
    """
    Load Unique_Source_Pol_Distance_Table.csv selecting columns by name.

    """
    df = pd.read_csv(path)

    missing = [c for c in PANOPOULOU_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in Panopoulou catalog: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )

    # Keep only the columns we actually need
    keep = PANOPOULOU_REQUIRED + [c for c in PANOPOULOU_DIST_COLS if c in df.columns]
    df = df[keep].copy()

    df = df.rename(columns={"EDR3_source_id": "GID", "Dec": "DEC"})

    df["GID"]    = df["GID"].astype(str)
    df["starID"] = df["starID"].astype(str)

    for col in ["RA", "DEC", "p", "e_p", "evpa", "e_evpa"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["RA", "DEC"]).reset_index(drop=True)
    return df



def load_merged_catalog(path: Path) -> pd.DataFrame:            #load your own observations
    """
    Load merged_output.csv using named columns 

    The three column names are controlled by the MERGED_*_COL constants at
    the top of the file — edit them there if your headers differ.

    """
    df = pd.read_csv(path)

    needed = [MERGED_NAME_COL, MERGED_RA_COL, MERGED_DEC_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in merged catalog: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )

    out = df[[MERGED_NAME_COL, MERGED_RA_COL, MERGED_DEC_COL]].copy()
    out = out.rename(columns={
        MERGED_NAME_COL: "obs_name",
        MERGED_RA_COL:   "obs_RA",
        MERGED_DEC_COL:  "obs_DEC",
    })

    out["obs_name"] = out["obs_name"].astype(str)
    out["obs_RA"]   = pd.to_numeric(out["obs_RA"],  errors="coerce")
    out["obs_DEC"]  = pd.to_numeric(out["obs_DEC"], errors="coerce")
    out = out.dropna(subset=["obs_RA", "obs_DEC"]).reset_index(drop=True)
    return out



# coordinate conversion and spatial filtering
def add_galactic_coordinates(df: pd.DataFrame) -> pd.DataFrame:                 
    """Append Galactic longitude (l) and latitude (b) columns in degrees."""
    coords = SkyCoord(
        ra=df["RA"].to_numpy() * u.deg,
        dec=df["DEC"].to_numpy() * u.deg,
    )
    out = df.copy()
    out["l"] = coords.galactic.l.deg
    out["b"] = coords.galactic.b.deg
    return out


def coarse_region_cut(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast rectangular pre-filter: 220° < l < 360°, 65° < b < 89°.
    Mirrors the notebook's initial box cut before the polygon test.

    """
    mask = (
        (df["l"] > 220.0) & (df["l"] < 360.0)
        & (df["b"] > 65.0) & (df["b"] < 89.0)
    )
    return df.loc[mask].copy().reset_index(drop=True)


def build_polygon(
    points_lb: Sequence[Tuple[float, float]],
) -> pygplates.PolygonOnSphere:
    """Construct a pygplates spherical polygon from (l, b) vertex pairs."""
    sphere_points = [pygplates.PointOnSphere(b, l) for l, b in points_lb]
    return pygplates.PolygonOnSphere(sphere_points)


def select_inside_polygon(
    df: pd.DataFrame,
    points_lb: Sequence[Tuple[float, float]],
) -> SelectionResult:
    """
    Retain only stars that fall inside the spherical polygon.
    Tags every selected row with origin = 'external_panopoulou_2025'.
    """
    polygon = build_polygon(points_lb)
    keep: List[bool] = []

    for l_val, b_val in zip(df["l"], df["b"]):
        point = pygplates.PointOnSphere(float(b_val), float(l_val))
        keep.append(bool(polygon.is_point_in_polygon(point.to_lat_lon())))

    selected = df.loc[keep].copy().reset_index(drop=True)
    selected["origin"] = "external_panopoulou_2025"
    return SelectionResult(selected=selected, polygon=list(points_lb))


# crossmatch
def crossmatch_with_observed(
    external_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    max_sep_arcsec: float = 3.0,
) -> pd.DataFrame:
    """
    For each of my observed stars find the nearest Panopoulou catalog star.

    Parameters----->
    external_df    : Panopoulou stars inside the polygon
    observed_df    : my merged_output stars
    max_sep_arcsec : maximum on-sky separation to count as a match

    Returns------->
    DataFrame with one row per observed star that found a match, containing
    columns from both catalogs plus sep_arcsec.
    """
    if external_df.empty or observed_df.empty:
        return pd.DataFrame()

    obs_coords = SkyCoord(
        ra=observed_df["obs_RA"].to_numpy() * u.deg,
        dec=observed_df["obs_DEC"].to_numpy() * u.deg,
    )
    ext_coords = SkyCoord(
        ra=external_df["RA"].to_numpy() * u.deg,
        dec=external_df["DEC"].to_numpy() * u.deg,
    )

    # Each observed star → its nearest Panopoulou neighbour
    idx, sep2d, _ = obs_coords.match_to_catalog_sky(ext_coords)

    matched_ext = external_df.iloc[idx].reset_index(drop=True)

    result = observed_df.reset_index(drop=True).copy()
    result["GID"]        = matched_ext["GID"]
    result["starID"]     = matched_ext["starID"]
    result["RA"]         = matched_ext["RA"]
    result["DEC"]        = matched_ext["DEC"]
    result["l"]          = matched_ext["l"]
    result["b"]          = matched_ext["b"]
    result["p"]          = matched_ext["p"]
    result["e_p"]        = matched_ext["e_p"]
    result["evpa"]       = matched_ext["evpa"]
    result["e_evpa"]     = matched_ext["e_evpa"]
    result["origin"]     = matched_ext["origin"]
    result["sep_arcsec"] = sep2d.arcsec

    # Carry distance columns if present
    for col in PANOPOULOU_DIST_COLS:
        if col in matched_ext.columns:
            result[col] = matched_ext[col].values

    result = result.loc[result["sep_arcsec"] <= max_sep_arcsec].copy()
    result = result.reset_index(drop=True)
    return result



def save_selection(df: pd.DataFrame, output_path: Path) -> None:
    """Save the full polygon selection with polarization and distance columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        ["GID", "starID", "RA", "DEC", "l", "b", "p", "e_p", "evpa", "e_evpa", "origin"]
        + [c for c in PANOPOULOU_DIST_COLS if c in df.columns]
    )
    df.to_csv(output_path, index=False, columns=[c for c in cols if c in df.columns])


def save_crossmatch(df: pd.DataFrame, output_path: Path) -> None:
    """Save the crossmatched table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        ["obs_name", "obs_RA", "obs_DEC",
         "GID", "starID", "RA", "DEC", "l", "b",
         "p", "e_p", "evpa", "e_evpa", "origin"]
        + [c for c in PANOPOULOU_DIST_COLS if c in df.columns]
        + ["sep_arcsec"]
    )
    df.to_csv(output_path, index=False, columns=[c for c in cols if c in df.columns])



def quick_plot(
    selected: pd.DataFrame,
    polygon: Sequence[Tuple[float, float]],
    diff_map_path: Path | None = None,
    show_polygon: bool = True,
) -> None:
    """
    Diagnostic scatter plot of the selected Panopoulou polygon stars.
    Uses a HEALPix dust-map background if healpy and the FITS file are available;
    falls back to a plain matplotlib scatter otherwise.
    """
    output_fig = Path("expanded_polygon_selection.png")
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    #plt.figure(figsize=(10, 10))           #don't need them whily using healpix
    used_healpy = False

    if diff_map_path is not None and diff_map_path.exists() and hp is not None:
        diff_map = hp.read_map(str(diff_map_path))
        hp.gnomview(
            diff_map,
            rot=[265, 80], min=-0.03, max=0.03, cmap="magma",
            xsize=150, ysize=150, fig=1, coord="G", reso=11,
            title="Expanded polygon selection", unit="diff", format="%.2g",
        )
        hp.graticule()
        hp.projscatter(
            selected["l"], selected["b"],
            marker="^", c="cyan", lonlat=True, coord="G",
        )
        if show_polygon:
            poly_l = [p[0] for p in polygon] + [polygon[0][0]]
            poly_b = [p[1] for p in polygon] + [polygon[0][1]]
            hp.projplot(poly_l, poly_b, lonlat=True, coord="G")
        used_healpy = True

    if not used_healpy:
        plt.scatter(selected["l"], selected["b"], marker="^", s=25,
                    label="Panopoulou polygon stars")
        if show_polygon:
            poly_l = [p[0] for p in polygon] + [polygon[0][0]]
            poly_b = [p[1] for p in polygon] + [polygon[0][1]]
            plt.plot(poly_l, poly_b, label="Polygon boundary")
            plt.legend()
        plt.xlabel("Galactic longitude l [deg]")
        plt.ylabel("Galactic latitude b [deg]")
        plt.title("Expanded polygon selection")
        plt.gca().invert_xaxis()

    #plt.tight_layout()         #don't need them whily using healpix
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_fig}")
    plt.show()



def choose_polygon(name: str) -> List[Tuple[float, float]]:
    presets = {"original": ORIGINAL_POLYGON, "expanded": EXPANDED_POLYGON}
    if name not in presets:
        raise ValueError(f"Unknown polygon preset '{name}'. Choose: {list(presets)}")
    return presets[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Panopoulou stars inside a sky polygon and "
            "crossmatch with your own observations from merged_output.csv."
        )
    )
    parser.add_argument(
        "--catalog", type=Path,
        default=Path("..") / "99_scripts_examples" / "in_agglomer" / "starpol_compilation-main" / "DataProducts" / "Unique_Source_Pol_Distance_Table.csv",
        help="Path to Unique_Source_Pol_Distance_Table.csv",
    )
    parser.add_argument(
        "--merged-catalog", type=Path,
        default=Path("..") / "2_sky_plot" / "merged_output.csv",
        help="Path to your observed merged_output.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("..") / "0_data" / "R" / "external_panopoulou_expanded_polygon.csv",
        help="Where to save the selected external stars",
    )
    parser.add_argument(
        "--match-output", type=Path,
        default=Path("external_panopoulou_crossmatched.csv"),
        help="Where to save the crossmatched pairs",
    )
    parser.add_argument(
        "--polygon", choices=["original", "expanded"], default="expanded",
        help="Polygon preset to use (default: expanded)",
    )
    parser.add_argument(
        "--diff-map", type=Path,
        default=Path("..") / "2_sky_plot" / "diff_ebv_gnilc_lenz.fits",
        help="Optional FITS dust map for the background plot",
    )
    parser.add_argument(
        "--max-sep-arcsec", type=float, default=3.0,
        help="Maximum angular separation for a crossmatch in arcsec (default: 3.0)",
    )
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True,
        help="Show/save a diagnostic plot (default: True; use --no-plot to skip)",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()
    polygon = choose_polygon(args.polygon)

    
    print(f"Loading Panopoulou catalog: {args.catalog}")
    catalog = load_panopoulou_catalog(args.catalog)
    print(f"  {len(catalog)} stars loaded")

    catalog = add_galactic_coordinates(catalog)
    catalog = coarse_region_cut(catalog)
    print(f"  {len(catalog)} stars after rectangular cut")

    result = select_inside_polygon(catalog, polygon)
    print(f"  {len(result.selected)} stars inside the '{args.polygon}' polygon")

    save_selection(result.selected, args.output)
    print(f"Polygon selection saved → {args.output}")

   
    print(f"\nLoading observed catalog: {args.merged_catalog}")
    observed = load_merged_catalog(args.merged_catalog)
    print(f"  {len(observed)} observed stars loaded")

    matched = crossmatch_with_observed(
        result.selected,
        observed,
        max_sep_arcsec=args.max_sep_arcsec,
    )
    print(f"  {len(matched)} crossmatched pairs (tolerance = {args.max_sep_arcsec}\")")

    save_crossmatch(matched, args.match_output)
    print(f"Crossmatch saved → {args.match_output}")

    if len(matched):
        print("\nMatched stars  (obs_name | sep\" | p ± e_p | evpa ± e_evpa):")
        for _, row in matched.iterrows():
            print(
                f"  {str(row['obs_name']):20s}  sep={row['sep_arcsec']:.2f}\"  "
                f"p={row['p']:.4f}±{row['e_p']:.4f}  "
                f"evpa={row['evpa']:.1f}±{row['e_evpa']:.1f} deg"
            )
    else:
        print("  No crossmatches found — check --max-sep-arcsec or sky coverage.")

    
    if args.plot:
        quick_plot(result.selected, result.polygon, diff_map_path=args.diff_map)


if __name__ == "__main__":
    main()

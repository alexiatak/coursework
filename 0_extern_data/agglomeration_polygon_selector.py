#!/usr/bin/env python3
"""
replacement for the agglomeration_test.ipynb notebook.

Purpose:

1. Read the Panopoulou et al. optical polarization compilation
2. Convert RA/Dec to Galactic coordinates
3. Keep only stars inside a predefined sky polygon
4. Save the selected external stars to a separate file

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pygplates
from astropy.coordinates import SkyCoord
from astropy import units as u

try:
    import healpy as hp
except Exception:  # pragma: no cover - plotting background is optional
    hp = None


# Original polygon from the last meaningful notebook cell.
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

# Slightly expanded version
EXPANDED_POLYGON = [
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


@dataclass
class SelectionResult:
    selected: pd.DataFrame
    polygon: List[Tuple[float, float]]


def load_panopoulou_catalog(path: Path) -> pd.DataFrame:
    """Load the unique-source polarization catalog used by the notebook."""
    df = pd.read_csv(path)

    required_columns = [df.columns[0], df.columns[1], df.columns[2], df.columns[3]]
    subset = df.loc[:, required_columns].copy()
    subset.columns = ["GID", "starID", "RA", "DEC"]

    subset["GID"] = subset["GID"].astype(str)
    subset["starID"] = subset["starID"].astype(str)
    subset["RA"] = pd.to_numeric(subset["RA"], errors="coerce")
    subset["DEC"] = pd.to_numeric(subset["DEC"], errors="coerce")
    subset = subset.dropna(subset=["RA", "DEC"]).reset_index(drop=True)
    return subset


def add_galactic_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    coords = SkyCoord(ra=df["RA"].to_numpy() * u.deg, dec=df["DEC"].to_numpy() * u.deg)
    out = df.copy()
    out["l"] = coords.galactic.l.deg
    out["b"] = coords.galactic.b.deg
    return out


def coarse_region_cut(df: pd.DataFrame) -> pd.DataFrame:
    """Match the broad rectangular filter used in the notebook."""
    mask = (
        (df["l"] > 220.0)
        & (df["l"] < 359.99)
        & (df["b"] > 65.0)
        & (df["b"] < 89.0)
    )
    return df.loc[mask].copy().reset_index(drop=True)


def build_polygon(points_lb: Sequence[Tuple[float, float]]) -> pygplates.PolygonOnSphere:
    sphere_points = [pygplates.PointOnSphere(b, l) for l, b in points_lb]
    return pygplates.PolygonOnSphere(sphere_points)


def select_inside_polygon(df: pd.DataFrame, points_lb: Sequence[Tuple[float, float]]) -> SelectionResult:
    polygon = build_polygon(points_lb)
    keep_rows: List[bool] = []

    for l_val, b_val in zip(df["l"], df["b"]):
        point = pygplates.PointOnSphere(float(b_val), float(l_val))
        keep_rows.append(bool(polygon.is_point_in_polygon(point.to_lat_lon())))

    selected = df.loc[keep_rows].copy().reset_index(drop=True)
    selected["origin"] = "external_panopoulou_2025"
    return SelectionResult(selected=selected, polygon=list(points_lb))


def save_selection(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["GID", "starID", "RA", "DEC", "l", "b", "origin"]
    df.to_csv(output_path, index=False, columns=cols)


def quick_plot(
    selected: pd.DataFrame,
    polygon: Sequence[Tuple[float, float]],
    diff_map_path: Path | None = None,
    show_polygon: bool = True,
) -> None:
    """Make a simple diagnostic plot just like the notebook."""
    #plt.figure(figsize=(10, 10))

    used_healpy_background = False
    if diff_map_path is not None and diff_map_path.exists() and hp is not None:
        diff_map = hp.read_map(str(diff_map_path))
        hp.gnomview(
            diff_map,
            rot=[265, 80],
            min=-0.03,
            max=0.03,
            cmap="magma",
            xsize=150,
            ysize=150,
            fig=1,
            coord="G",
            reso=11,
            title="Expanded polygon selection",
            unit="diff",
            format="%.2g",
        )
        hp.graticule()
        hp.projscatter(selected["l"], selected["b"], marker="^", c="cyan", lonlat=True, coord="G")
        if show_polygon:
            poly_l = [p[0] for p in polygon] + [polygon[0][0]]
            poly_b = [p[1] for p in polygon] + [polygon[0][1]]
            hp.projplot(poly_l, poly_b, lonlat=True, coord="G")
        used_healpy_background = True

    if not used_healpy_background:
        plt.figure(figsize=(10, 10)) 
        plt.scatter(selected["l"], selected["b"], marker="^", s=25)
        if show_polygon:
            poly_l = [p[0] for p in polygon] + [polygon[0][0]]
            poly_b = [p[1] for p in polygon] + [polygon[0][1]]
            plt.plot(poly_l, poly_b)
        plt.xlabel("Galactic longitude l [deg]")
        plt.ylabel("Galactic latitude b [deg]")
        plt.title("Expanded polygon selection")
        plt.gca().invert_xaxis()
        plt.tight_layout()

    
    plt.savefig(r"./expanded_polygon_selection.png", dpi=300, bbox_inches="tight")
    plt.show()


def choose_polygon(name: str) -> List[Tuple[float, float]]:
    if name == "original":
        return ORIGINAL_POLYGON
    if name == "expanded":
        return EXPANDED_POLYGON
    raise ValueError(f"Unknown polygon preset: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select external polarization stars inside a sky polygon."
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(r"../99_scripts_examples/in_agglomer/starpol_compilation-main/DataProducts/Unique_Source_Pol_Distance_Table.csv"),
        help="Path to Unique_Source_Pol_Distance_Table.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"./external_panopoulou_expanded_polygon.csv"),
        help="Where to save the selected external stars as CSV",
    )

    parser.add_argument(
        "--polygon",
        choices=["original", "expanded"],
        default="expanded",
        help="Polygon preset to use",
    )

    parser.add_argument(
        "--diff-map",
        type=Path,
        default=Path(r"../2_sky_plot/diff_ebv_gnilc_lenz.fits"),
        help="Optional path to diff_ebv_gnilc_lenz.fits for background plot",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Show a diagnostic plot (default: True)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    polygon = choose_polygon(args.polygon)
    catalog = load_panopoulou_catalog(args.catalog)
    catalog = add_galactic_coordinates(catalog)
    catalog = coarse_region_cut(catalog)
    result = select_inside_polygon(catalog, polygon)
    save_selection(result.selected, args.output)

    print(f"Saved {len(result.selected)} selected stars to {args.output}")
    print(f"Polygon preset: {args.polygon}")

    if args.plot:
        quick_plot(result.selected, result.polygon, diff_map_path=args.diff_map)


if __name__ == "__main__":
    main()

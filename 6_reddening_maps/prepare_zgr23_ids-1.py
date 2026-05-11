"""
prepare_zgr23_ids.py

Produces two one-column CSV files of Gaia source_ids for the TOPCAT ZGR23 query:
  - zgr23_query_robopol.csv
  - zgr23_query_panopoulou.csv

Run once from the repo root. Outputs go to the same directory as this script
(or change OUTPUT_DIR below).
"""

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
MERGED_OUTPUT      = "../2_sky_plot/merged_output.csv"
MERGED_WITH_GAIAID = "../4_gaia/match_with_gaia/merged_with_gaiaid.csv"
PANOPOULOU         = "../0_data/R/external_panopoulou_expanded_polygon.csv"
OUTPUT_DIR         = "../6_reddening_maps/"

# ── RoboPol ────────────────────────────────────────────────────────────────────
robopol = pd.read_csv(MERGED_OUTPUT)
gaia    = pd.read_csv(MERGED_WITH_GAIAID)

# keep all rows with a valid gaia_id
gaia_clean = gaia[["Name", "gaia_id"]].dropna(subset=["gaia_id"])

# gaia_id can come in as float if there were NaNs; force to int64
gaia_clean["gaia_id"] = gaia_clean["gaia_id"].astype("int64")

merged = robopol[["Name"]].merge(gaia_clean, on="Name", how="inner")

print(f"RoboPol: {len(robopol)} stars in merged_output, "
      f"{len(merged)} with a Gaia ID.")

out_robopol = merged[["gaia_id"]].rename(columns={"gaia_id": "source_id"})
out_robopol.to_csv(OUTPUT_DIR + "zgr23_query_robopol.csv", index=False)
print(f"  -> wrote {OUTPUT_DIR}zgr23_query_robopol.csv")

# ── Panopoulou ─────────────────────────────────────────────────────────────────
pano = pd.read_csv(PANOPOULOU)
out_pano = pano[["GID"]].rename(columns={"GID": "source_id"})
out_pano.to_csv(OUTPUT_DIR + "zgr23_query_panopoulou.csv", index=False)
print(f"Panopoulou: {len(out_pano)} stars.")
print(f"  -> wrote {OUTPUT_DIR}zgr23_query_panopoulou.csv")

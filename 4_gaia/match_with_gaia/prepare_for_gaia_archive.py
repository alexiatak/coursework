#!/usr/bin/env python3
"""

Build the upload CSV in exactly this format :

    Name,ra,dec,gid
    Mark_0,194.58873,21.12631,3942609523362045312
    ...

This is what you upload to https://gea.esac.esa.int/archive/ as a user table,
then run his ADQL query against it.

Run AFTER match_gaia_ids.py and patch_dima_gaia_ids.py.
"""

import pandas as pd

INPUT  = "merged_with_gaiaid.csv"
OUTPUT = "for_gaia_archive.csv"

df = pd.read_csv(INPUT, dtype={"gaia_id": str})
df["gaia_id"] = df["gaia_id"].fillna("").astype(str)

# Keep only matched rows (those with a Gaia ID)
ok = df[df["matched"] & (df["gaia_id"] != "")].copy()

# Rename to match Dima's exact column scheme
out = ok[["Name", "ra", "dec", "gaia_id"]].rename(columns={"gaia_id": "gid"})
out.to_csv(OUTPUT, index=False)

print(f"Wrote {OUTPUT} with {len(out)} rows")
print(f"\nFirst few rows (the format Dima asked for):")
print(out.head().to_string(index=False))
print(f"\nUpload this file to https://gea.esac.esa.int/archive/")
print(f"as a user table called 'mark', then run Dima's ADQL query:")
print()
print("SELECT mark.Name, mark.ra, mark.dec, mark.gid,")
print("       edr3.r_med_photogeo, edr3.r_lo_photogeo, edr3.r_hi_photogeo")
print("FROM   external.gaiaedr3_distance AS edr3")
print("JOIN   user_<your_username>.mark   AS mark")
print("  ON   mark.gid = edr3.source_id")
print()
print(f"save as bj_distances.csv in 4_gaia")
print()


#!/usr/bin/env python3
"""

add the 13 Gaia IDs Dima looked up by hand into merged_with_gaiaid.csv.

Run after match_gaia_ids.py.
"""

import pandas as pd

INPUT  = "merged_with_gaiaid.csv"
OUTPUT = "merged_with_gaiaid.csv"     # overwrite in place

# Manually supplied by Dima (email 23 Apr 2026)
DIMA_IDS = {
    "Mark_0":  "3942609523362045312",
    "Mark_1":  "3942777435107365632",
    "Mark_2":  "3942782387205873536",
    "Mark_3":  "3942619556405652608",
    "Mark_4":  "3944050639508731264",
    "Mark_5":  "3942750737590666624",
    "Mark_7":  "3954771530713595008",
    "Mark_8":  "3949433161243432704",
    "Mark_9":  "3949432542768145280",
    "Mark_10": "3925221017550590208",
    "Mark_11": "3926214224444021760",
    "Mark_12": "3972586367863070208",
    "Mark_13": "3972526577623362560",
}

df = pd.read_csv(INPUT, dtype={"gaia_id": str})
# Defensive: a freshly-read CSV may have NaN in gaia_id where it was empty.
df["gaia_id"] = df["gaia_id"].fillna("").astype(str)

n_patched = 0
for name, gid in DIMA_IDS.items():
    mask = df["Name"] == name
    if mask.sum() == 0:
        print(f"WARNING: {name} not found in {INPUT}, skipping")
        continue
    df.loc[mask, "gaia_id"] = gid
    df.loc[mask, "matched"] = True   # promote them to "matched"
    n_patched += int(mask.sum())

df.to_csv(OUTPUT, index=False)
print(f"Patched {n_patched} rows. New total matched: {df['matched'].sum()}")

# Also regenerate gaia_id_list.csv with the new IDs included
ok_ids = (df.loc[df["matched"], "gaia_id"]
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True))
ok_ids.to_frame(name="source_id").to_csv("gaia_id_list.csv", index=False)
print(f"Updated gaia_id_list.csv: {len(ok_ids)} unique IDs")

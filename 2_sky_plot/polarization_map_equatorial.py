import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

FITS_FILE   = "./diff_ebv_gnilc_lenz.fits"
MY_STARS    = "./merged_output.csv"
EXT_CATALOG = "../0_data/R/external_panopoulou_expanded_polygon.csv"
OUTPUT_FILE = "./polarization_map_equatorial.png"

SCALE_MY  = 0.4
SCALE_EXT = 1.5

dust_map = hp.read_map(FITS_FILE)
print(f"Map loaded: nside={hp.npix2nside(dust_map.size)}")

df_my = pd.read_csv(MY_STARS)
df_my = df_my.dropna(subset=["PA[deg]", "P[%]", "ra", "dec"]).reset_index(drop=True)
N_my  = len(df_my)
print(f"My stars loaded: {N_my}")

df_ext     = pd.read_csv(EXT_CATALOG)
df_ext_seg = df_ext.dropna(subset=["evpa", "p"]).reset_index(drop=True)
df_ext_dot = df_ext[df_ext["evpa"].isna()].reset_index(drop=True)
N_ext      = len(df_ext_seg)
print(f"Panopoulou: {N_ext} with evpa, {len(df_ext_dot)} without")

def seg_endpoints(ra, dec, pa_deg, half_deg):
    pa   = np.radians(pa_deg)
    ras  = [ra  - half_deg * np.sin(pa),  ra  + half_deg * np.sin(pa)]
    decs = [dec - half_deg * np.cos(pa),  dec + half_deg * np.cos(pa)]
    return ras, decs

#fig = plt.figure(figsize=(14, 10))

hp.gnomview(
    dust_map,
    rot=[185, 16, 0],
    coord=["G", "C"],
    min=-0.03, max=0.03,
    cmap="magma",
    xsize=300, ysize=300,
    reso=6,
    #fig=fig.number,
    title="Markkanen cloud — equatorial frame",
    unit="E(B-V) diff",
    format="%.2g",
    notext=False,
)

fig = plt.gcf()
fig.set_size_inches(14, 10)

hp.graticule(dpar=5, dmer=5, alpha=0.55, color="white", lw=0.6)

for idx, row in df_my.iterrows():
    half = SCALE_MY * row["P[%]"]
    ras, decs = seg_endpoints(row["ra"], row["dec"], row["PA[deg]"], half)
    hp.projplot(ras, decs, lonlat=True, coord="C",
                color="darkgreen", lw=1.4,
                label="My stars" if idx == 0 else None)

for idx, row in df_ext_seg.iterrows():
    P_pct = row["p"] * 100.0
    half  = SCALE_EXT * P_pct
    evpa = row["evpa"]% 180.0 
    ras, decs = seg_endpoints(row["RA"], row["DEC"], evpa, half)
    hp.projplot(ras, decs, lonlat=True, coord="C",
                color="cyan", lw=1.0,
                label="Panopoulou+2025" if idx == 0 else None)

if len(df_ext_dot) > 0:
    hp.projscatter(df_ext_dot["RA"].values, df_ext_dot["DEC"].values,
                   lonlat=True, coord="C",
                   marker="o", s=14, color="cyan", alpha=0.5)

leg_my  = mlines.Line2D([], [], color="darkgreen", lw=1.8,
                         label=f"My stars (N={N_my})")
leg_ext = mlines.Line2D([], [], color="cyan", lw=1.5,
                         label=f"Panopoulou+2025 (N={N_ext})")
handles = [leg_my, leg_ext]
if len(df_ext_dot) > 0:
    leg_dot = mlines.Line2D([], [], color="cyan", lw=0,
                             marker="o", markersize=5, alpha=0.6,
                             label=f"Panopoulou no EVPA (N={len(df_ext_dot)})")
    handles.append(leg_dot)

plt.legend(handles=handles, loc="upper right", framealpha=0.85, fontsize=10)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUTPUT_FILE}")
plt.close()

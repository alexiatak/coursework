import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.coordinates import SkyCoord
from astropy import units as u

FITS_FILE   = "./diff_ebv_gnilc_lenz.fits"
MY_STARS    = "./merged_output.csv"
EXT_CATALOG = "../0_data/R/external_panopoulou_expanded_polygon.csv"
OUTPUT_FILE = "./polarization_map_equatorial.png"

SCALE_MY  = 0.5
SCALE_EXT = 0.5

# reference-bar configuration 
REF_P_PERCENT   = 1.0                              # reference polarization value
REF_HALF_DEG    = SCALE_MY * REF_P_PERCENT         # half-length in degrees
MAP_ROT_RA      = 185.0                            # equatorial centre used in rot
MAP_ROT_DEC     = 16.0

# Turn to true to test the evpa 
EVPA_TEST_MODE = True

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
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    p1 = c.directional_offset_by(pa_deg*u.deg,          half_deg*u.deg)
    p2 = c.directional_offset_by((pa_deg+180.0)*u.deg,  half_deg*u.deg)
    return [p1.ra.deg, p2.ra.deg], [p1.dec.deg, p2.dec.deg]


#fig = plt.figure(figsize=(14, 10))

hp.gnomview(
    dust_map,
    rot=[MAP_ROT_RA, MAP_ROT_DEC, 0],
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

#  1% reference bar
# length is the fraction of figure width that a 2*REF_HALF_DEG bar occupies
# at the map centre:  map_span = xsize_px * reso_arcmin / 60  (degrees)
ref_full_deg = 2.0 * REF_HALF_DEG
map_span_deg = (300 * 6) / 60.0              # xsize=300, reso=6 arcmin
bar_frac     = ref_full_deg / map_span_deg
ax = plt.gca()
x0, y0 = 0.06, 0.08                          # lower-left corner of bar (axes frac)

ax.plot([x0, x0 + bar_frac], [y0, y0],
        color="black", lw=4.0,
        transform=ax.transAxes, solid_capstyle="butt", zorder=10)
ax.plot([x0, x0 + bar_frac], [y0, y0],
        color="white", lw=2.0,
        transform=ax.transAxes, solid_capstyle="butt", zorder=11)
ax.text(x0 + bar_frac / 2.0, y0 - 0.025,
        f"P = {REF_P_PERCENT:.0f}%",
        color="white", fontsize=10, ha="center", va="top",
        transform=ax.transAxes, zorder=11)

#  EVPA verification test mode
if EVPA_TEST_MODE:
    test_rows = []

    idx_my  = [0, N_my  // 2, N_my  - 1]
    idx_ext = [0, N_ext // 2, N_ext - 1]

    counter = 0
    for i in idx_my:
        counter += 1
        row = df_my.iloc[i]
        ra_c, dec_c = float(row["ra"]), float(row["dec"])
        evpa_in = float(row["PA[deg]"])
        P_pct   = float(row["P[%]"])
        half    = SCALE_MY * P_pct
        ras, decs = seg_endpoints(ra_c, dec_c, evpa_in, half)
        # thicker red line, drawn last so it sits on top
        hp.projplot(ras, decs, lonlat=True, coord="C",
                    color="red", lw=2.8)
        hp.projtext(ras[1], decs[1], f" {counter}",
                    lonlat=True, coord="C",
                    color="red", fontsize=11, fontweight="bold")
        mean_dec_rad = np.radians(0.5 * (decs[0] + decs[1]))
        dra  = (ras[1]  - ras[0])  * np.cos(mean_dec_rad)
        ddec = (decs[1] - decs[0])
        evpa_rec = np.degrees(np.arctan2(dra, ddec)) % 180.0
        test_rows.append({
            "Idx":     counter,
            "Source":  "my",
            "Name":    str(row["Name"]),
            "RA":      round(ra_c, 4),
            "Dec":     round(dec_c, 4),
            "P[%]":    round(P_pct, 3),
            "EVPA_in": round(evpa_in % 180.0, 3),
            "ra1":     round(ras[0],  4),
            "dec1":    round(decs[0], 4),
            "ra2":     round(ras[1],  4),
            "dec2":    round(decs[1], 4),
            "EVPA_rec": round(evpa_rec, 3),
        })

    # Panopoulou catalog may or may not carry a name-like column
    pan_name_col = None
    for candidate in ("Name", "name", "ID", "id", "GaiaDR3"):
        if candidate in df_ext_seg.columns:
            pan_name_col = candidate
            break

    for i in idx_ext:
        counter += 1
        row = df_ext_seg.iloc[i]
        ra_c, dec_c = float(row["RA"]), float(row["DEC"])
        evpa_in = float(row["evpa"]) % 180.0
        P_pct   = float(row["p"]) * 100.0
        half    = SCALE_EXT * P_pct
        ras, decs = seg_endpoints(ra_c, dec_c, evpa_in, half)
        hp.projplot(ras, decs, lonlat=True, coord="C",
                    color="orange", lw=2.8)
        hp.projtext(ras[1], decs[1], f" {counter}",
                    lonlat=True, coord="C",
                    color="orange", fontsize=11, fontweight="bold")
        mean_dec_rad = np.radians(0.5 * (decs[0] + decs[1]))
        dra  = (ras[1]  - ras[0])  * np.cos(mean_dec_rad)
        ddec = (decs[1] - decs[0])
        evpa_rec = np.degrees(np.arctan2(dra, ddec)) % 180.0
        if pan_name_col is not None:
            name_val = str(row[pan_name_col])
        else:
            name_val = f"row{i}"
        test_rows.append({
            "Idx":     counter,
            "Source":  "Panopoulou",
            "Name":    name_val,
            "RA":      round(ra_c, 4),
            "Dec":     round(dec_c, 4),
            "P[%]":    round(P_pct, 3),
            "EVPA_in": round(evpa_in, 3),
            "ra1":     round(ras[0],  4),
            "dec1":    round(decs[0], 4),
            "ra2":     round(ras[1],  4),
            "dec2":    round(decs[1], 4),
            "EVPA_rec": round(evpa_rec, 3),
        })

    test_df = pd.DataFrame(test_rows)
    print("\n=== EVPA verification test table ===")
    print(test_df.to_string(index=False))
    print("===================================\n")

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
if EVPA_TEST_MODE:
    handles.append(mlines.Line2D([], [], color="red",    lw=2.5, label="My test stars"))
    handles.append(mlines.Line2D([], [], color="orange", lw=2.5, label="Panopoulou test stars"))

plt.legend(handles=handles, loc="upper right", framealpha=0.85, fontsize=10)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved -> {OUTPUT_FILE}")
plt.close()

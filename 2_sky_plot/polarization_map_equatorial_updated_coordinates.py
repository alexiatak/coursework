import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy.coordinates import SkyCoord
from astropy import units as u

FITS_FILE   = "./diff_ebv_gnilc_lenz.fits"
MY_STARS    = "./merged_output.csv"
#EXT_CATALOG = "../0_data/R/external_panopoulou_expanded_polygon.csv"
EXT_CATALOG = "../0_data/R/external_panopoulou_combined_polygon.csv"
OUTPUT_FILE = "./polarization_map_equatorial_smaller_polygon.png"

SCALE_MY  = 0.5
SCALE_EXT = 0.5

# reference-bar configuration 
REF_P_PERCENT   = 1.0                              # reference polarization value
REF_HALF_DEG    = SCALE_MY * REF_P_PERCENT         # half-length in degrees
MAP_ROT_RA      = 185.0                            # equatorial centre used in rot
MAP_ROT_DEC     = 16.0

# Manual Equatorial coordinate labels for the clean plot frame.
# Keep only the map-centre coordinates and one extra coordinate on each axis.
RA_LABEL_CENTER  = MAP_ROT_RA
RA_LABEL_EXTRA   = 195.0
DEC_LABEL_CENTER = MAP_ROT_DEC
DEC_LABEL_EXTRA  = 5.0

# Turn to true to test the evpa 
EVPA_TEST_MODE = False

dust_map = hp.read_map(FITS_FILE)
print(f"Map loaded: nside={hp.npix2nside(dust_map.size)}")

df_my = pd.read_csv(MY_STARS)
df_my = df_my.dropna(subset=["PA[deg]", "P[%]", "ra", "dec"]).reset_index(drop=True)
N_my  = len(df_my)
print(f"Robopol loaded: {N_my}")

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
    title="",
    unit=r"$\Delta E(B-V)$",
    format="%.2g",
    notext=True,
)

fig = plt.gcf()
fig.set_size_inches(14, 10)

hp.graticule(dpar=5, dmer=5, alpha=0.55, color="white", lw=0.6)

# ------------------------------------------------------------
# Manual Equatorial coordinate labels on the top and left frame
# ------------------------------------------------------------
fig_cur = plt.gcf()
ax = plt.gca()
ax_bbox = ax.get_position()

map_span_deg = 300 * 6 / 60.0
half_deg = map_span_deg / 2.0

# Top right-ascension labels.
# In healpy celestial projections, RA increases to the left.
y_top = ax_bbox.y1 + 0.005
x_ra_center = (ax_bbox.x0 + ax_bbox.x1) / 2.0
x_ra_extra = x_ra_center - ((RA_LABEL_EXTRA - RA_LABEL_CENTER) / (2.0 * half_deg)) * (ax_bbox.x1 - ax_bbox.x0)

fig_cur.text(x_ra_extra, y_top, f"{RA_LABEL_EXTRA:.1f}°",
             ha="center", va="bottom", fontsize=12)

fig_cur.text(x_ra_center, y_top, f"{RA_LABEL_CENTER:.1f}°",
             ha="center", va="bottom", fontsize=12)

# Left declination labels.
x_left = ax_bbox.x0 - 0.005
y_dec_center = (ax_bbox.y0 + ax_bbox.y1) / 2.0
y_dec_extra = y_dec_center + ((DEC_LABEL_EXTRA - DEC_LABEL_CENTER) / (2.0 * half_deg)) * (ax_bbox.y1 - ax_bbox.y0)

fig_cur.text(x_left, y_dec_extra, f"{DEC_LABEL_EXTRA:.1f}°",
             ha="right", va="center", fontsize=12)

fig_cur.text(x_left, y_dec_center, f"{DEC_LABEL_CENTER:.1f}°",
             ha="right", va="center", fontsize=12)

# Axis names.
fig_cur.text((ax_bbox.x0 + ax_bbox.x1) / 2.0, y_top + 0.03,
             r"$RA$ [deg]",
             ha="center", va="bottom", fontsize=18)

fig_cur.text(x_left - 0.035, y_dec_center + 0.2,
             r"$Dec$ [deg]",
             ha="center", va="center", rotation=90, fontsize=18)
# End of manual Equatorial axis labels

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

# 1% reference bar, drawn in sky coords with the same great-circle
# geometry as the star vectors, so it's directly comparable by eye
# to any nearby vector (both get the same local projection stretch).
ref_full_deg = 2.0 * REF_HALF_DEG
map_span_deg = (300 * 6) / 60.0                    # xsize=300, reso=6 arcmin

# Anchor the bar in the lower-left of the field
ref_ra  = MAP_ROT_RA  + 0.40 * map_span_deg / np.cos(np.radians(MAP_ROT_DEC - 0.40 * map_span_deg))
ref_dec = MAP_ROT_DEC - 0.40 * map_span_deg

# Horizontal bar: PA = 90 deg means aligned East-West on the sky.
ref_ras, ref_decs = seg_endpoints(ref_ra, ref_dec, pa_deg=90.0, half_deg=REF_HALF_DEG)

# Black outline first, white core on top.
hp.projplot(ref_ras, ref_decs, lonlat=True, coord="C",
            color="black", lw=5.0, zorder=10)
hp.projplot(ref_ras, ref_decs, lonlat=True, coord="C",
            color="white", lw=2.5, zorder=11)
hp.projtext(ref_ra, ref_dec - 0.6, f"P = {REF_P_PERCENT:.0f}%",
            lonlat=True, coord="C",
            color="white", fontsize=10, ha="center", va="top", zorder=11)

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
                         label=f"Robopol (N={N_my})")
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

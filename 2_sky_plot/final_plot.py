import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from pylab import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import pygplates
import random
import pandas as pd

from uncertainties import ufloat
from uncertainties import umath
from scipy import optimize, interpolate
from scipy.special import erf
from scipy.optimize import minimize
import scipy.integrate as integrate
import aplpy
from astropy import stats
from astropy.io import fits


SHOW_EXTERNAL_CATALOG = True        #   set to True to overlay the external Panopoulou catalog on the
                                    #   polarization map, or False to show only your own stars.


# Path to the external catalog CSV produced by agglomeration_polygon_selector.py
EXTERNAL_CATALOG_PATH = "../0_extern_data/external_panopoulou_expanded_polygon.csv"



diff_map = hp.read_map("./diff_ebv_gnilc_lenz.fits")

# reading the external/agglomeration list
targ_done_l = []
targ_done_b = []
targ_remaining_l = []
targ_remaining_b = []
remaining_names = []
fop = open('./Markkanen_schedule.csv')
for line in fop.readlines():
    if line.startswith("#"):
        continue
    sl = line.split(',')
    ra = float(sl[1])
    dec = float(sl[2])
    remaining_names.append(sl[0])
    c = SkyCoord(ra, dec, unit="deg", frame="icrs")
    targ_remaining_l.append(c.galactic.l.degree)
    targ_remaining_b.append(c.galactic.b.degree)
fop.close()


# reading the external/agglomeration list
targ_all_l = []
targ_all_b = []
fop = open('./Markkanen_schedule_original.csv')
for line in fop.readlines():
    if line.startswith("#"):
        continue
    sl = line.split(',')
    ra = float(sl[1])
    dec = float(sl[2])
    name = sl[0]
    c = SkyCoord(ra, dec, unit="deg", frame="icrs")
    targ_all_l.append(c.galactic.l.degree)
    targ_all_b.append(c.galactic.b.degree)
    if not name in remaining_names:
        targ_done_l.append(c.galactic.l.degree)
        targ_done_b.append(c.galactic.b.degree)
fop.close()

f = plt.figure(figsize=(12,12))

hp.gnomview(diff_map, rot=[265, 80], min=-0.03, max=0.03, cmap='magma',
            xsize=150, ysize=150, fig=1,
            coord='G', reso=11, title="GnomView", unit="diff", format="%.2g")
hp.graticule()

hp.projscatter(targ_all_l, targ_all_b, marker='.', c='g', lonlat=True, coord='G', label='remaining')
hp.projscatter(targ_done_l, targ_done_b, marker='.', c='k', lonlat=True, coord='G', label='observed')
plt.legend()
plt.savefig('mark_observed_targets.png')
plt.clf()


def appenzeller(l, b, evpa):
    '''Convert evpa to Galactic angle as in Appenzeller 1968
        Input:
            - l: list/array of galactic longitudes in degrees
            - b: list/array of galactic latitudes in degrees
            - evpa: list/array of EVPA in degrees
        Output:
            - array of angles with respect to NGP in degrees
    '''
    lp = 122.93200023
    bp = 27.12843
    targ_done_l = np.array(l)
    targ_done_b = np.array(b)
    evpa = np.array(evpa)
    Dt = np.arctan2(np.sin(np.radians(lp - targ_done_l)),
                    (np.cos(np.radians(targ_done_b)) * np.tan(np.radians(bp)) -
                     np.cos(np.radians(lp - targ_done_l)) * np.sin(np.radians(targ_done_b))))
    return evpa + Dt


################################
#   second part – polarization strip classes
################################

class star:
    def __init__(self, PD, PDerr, PA, PAerr):
        self.PD    = PD
        self.PDerr = PDerr
        self.PA    = PA
        self.PAerr = PAerr

class Obs():
    def __init__(self, name, JD, q, qerr, u, uerr, x=0.0, y=0.0):
        self.name = name
        self.JD   = JD
        self.q    = ufloat(q, qerr)
        self.u    = ufloat(u, uerr)
        self.x    = x
        self.y    = y

    def getP(self):
        return umath.sqrt(self.q ** 2 + self.u ** 2)

    def getPA(self):
        PAval = 0.5 * umath.atan2(self.u, self.q)
        PAunc = self.getSigma(self.getP().n, self.getP().s)
        return ufloat(np.degrees(PAval.n), np.degrees(PAunc))

    def correctInst(self, q_inst, u_inst):
        self.q = self.q - q_inst
        self.u = self.u - u_inst

    def correctRot(self, dPA):
        THETA = umath.radians(dPA)
        Qnew = self.q * umath.cos(2 * THETA) - self.u * umath.sin(2 * THETA)
        Unew = self.q * umath.sin(2 * THETA) + self.u * umath.cos(2 * THETA)
        self.q = Qnew
        self.u = Unew

    def EVPA_pdf(self, theta, P0):
        g = 1 / np.sqrt(np.pi)
        ita0 = float(P0) / np.sqrt(2) * np.cos(2 * theta)
        g = g * (g + ita0 * np.exp(ita0**2) * (1 + erf(ita0)))
        g = g * np.exp(-(float(P0)**2) / 2)
        return g

    def int_eq(self, sigma, snr):
        integ = integrate.quad(lambda x: self.EVPA_pdf(x, snr), -sigma, sigma)
        return abs(integ[0] - 0.68268949)

    def getSigma(self, pd, pd_err):
        snr = pd / pd_err
        if snr > 20:
            return 0.5 * 1.0 / float(snr)
        if snr < np.sqrt(2.0):
            pd = 0.0
        else:
            pd = np.sqrt(pd**2 - pd_err**2)
        snr = pd / pd_err
        res = minimize(self.int_eq, [np.pi / 50], args=(snr,), method='Nelder-Mead', tol=1e-5)
        if res.status != 0:
            print('Something is wrong with the EVPA uncertainty calculation:\n')
            return np.nan
        return res.x[0]

class Stand():
    def __init__(self, st):
        self.name = st
        stand = stand_pol.STANDARDS[self.name]
        self.PD    = ufloat(stand.PD / 100., stand.PDerr / 100.)
        self.PA    = ufloat(stand.PA, stand.PAerr)
        self.calcQU()

    def calcQU(self):
        self.q = self.PD * umath.cos(2 * umath.radians(self.PA))
        self.u = self.PD * umath.sin(2 * umath.radians(self.PA))


##################
# Merge my observations with coordinates

df_dat = pd.read_csv("../0_data/R/Markkanen_final_merged.dat", sep=r"\s+", engine="python")
df_dat.rename(columns=lambda x: x.lstrip('#'), inplace=True)

df_csv = pd.read_csv("Markkanen_schedule_original.csv")
df_csv.rename(columns={"#name": "Name"}, inplace=True)

df_csv_selected = df_csv[["Name", "ra", "dec"]]
df_merged = pd.merge(df_dat, df_csv_selected, on="Name", how="left")
df_merged.to_csv("merged_output.csv", index=False)

########################################################

def coordinates_os_stars():
    ra, dec, Ps, angles, l_for_angle, b_for_angle = [], [], [], [], [], []

    for _, row in df_merged.iterrows():
        x = row["ra"]
        y = row["dec"]
        PD = row["P[%]"]
        EVPA = row["PA[deg]"]

        ra.append(x)
        dec.append(y)
        Ps.append(PD)
        c = SkyCoord(ra[-1], dec[-1], unit="deg", frame="icrs")
        l_for_angle.append(c.galactic.l.degree)
        b_for_angle.append(c.galactic.b.degree)
        np.set_printoptions(legacy='1.25')
        evpa_corr = appenzeller(l_for_angle[-1], b_for_angle[-1], EVPA)
        angles.append(evpa_corr)


    angles = [a % 180.0 for a in angles]
    return ra, dec, Ps, angles

stars = coordinates_os_stars()




def coordinates_external_stars(catalog_path=EXTERNAL_CATALOG_PATH):
    """
    Load the Panopoulou polygon catalog and return plotting-ready arrays.

    Parameters
    ----------
    catalog_path : str
        Path to external_panopoulou_expanded_polygon.csv

    Returns
    -------
    ra, dec : lists of float  – ICRS coordinates in degrees
    Ps      : list of float  – polarization degree in % (same units as my stars)
    angles  : list of float  – corrected PA in degrees, range [0, 180)
    """
    df_ext = pd.read_csv(catalog_path)

    # Drop rows without measured polarization or angle
    df_ext = df_ext.dropna(subset=["p", "evpa"]).reset_index(drop=True)

    ra, dec, Ps, angles = [], [], [], []

    for _, row in df_ext.iterrows():
        x    = row["RA"]
        y    = row["DEC"]
        PD   = row["p"] * 100.0   
        EVPA = row["evpa"]         

        ra.append(x)
        dec.append(y)
        Ps.append(PD)

        c = SkyCoord(x, y, unit="deg", frame="icrs")
        l = c.galactic.l.degree
        b = c.galactic.b.degree

        evpa_corr = appenzeller(l, b, EVPA)
        angles.append(evpa_corr)

    
    angles = [a % 180.0 for a in angles]

    print(f"[external catalog] loaded {len(ra)} stars with valid p & evpa")
    return ra, dec, Ps, angles



def segments_on_map(fitsfile, ra, dec, Ps, pas,
                    scale=1000,
                    ext_ra=None, ext_dec=None, ext_Ps=None, ext_pas=None,
                    ext_scale=None):
    """
    Plot polarization segments on a HEALPix dust map.

    Parameters
    ----------
    fitsfile          : str   – path to the HEALPix FITS file
    ra, dec           : lists – ICRS coordinates of my stars (degrees)
    Ps                : list  – polarization degree in % for my stars
    pas               : array – corrected angles in radians for my stars
    scale             : float – segment half-length scale for my stars

    ext_ra, ext_dec   : lists  – ICRS coords of EXTERNAL stars (degrees);
                                 only used when SHOW_EXTERNAL_CATALOG is True
    ext_Ps            : list   – polarization degree in % for EXTERNAL stars
    ext_pas           : array  – corrected angles in radians for EXTERNAL stars
    ext_scale         : float  – segment half-length scale for EXTERNAL stars;
                                 defaults to `scale` if None
    """
    if ext_scale is None:
        ext_scale = scale

    mapread = hp.read_map(fitsfile)
    fig = plt.figure(figsize=(12, 12))
    hp.gnomview(mapread, rot=[265, 80], min=-0.03, max=0.03, cmap='magma',
                xsize=150, ysize=150, fig=1,
                coord='G', reso=11, title="GnomView", unit="diff", format="%.2g")
    hp.graticule()

    # ── my own stars (green) ────────────────────────────────────────────────
    for iv in range(len(pas)):
        c = SkyCoord(ra[iv], dec[iv], unit="deg", frame="icrs")
        l = c.galactic.l.degree
        b = c.galactic.b.degree
        linelength_half = scale * Ps[iv]
        bs = [b - linelength_half * np.cos(pas[iv]),
              b + linelength_half * np.cos(pas[iv])]
        ls = [l + linelength_half * np.sin(pas[iv]),
              l - linelength_half * np.sin(pas[iv])]
        hp.projplot(ls, bs, c='g', lonlat=True, coord='G',
                    label='My stars' if iv == 0 else None)

    # ── external Panopoulou stars (cyan) ─────────────────────────────────────
    if (SHOW_EXTERNAL_CATALOG and
            ext_ra is not None and ext_dec is not None
            and ext_Ps is not None and ext_pas is not None):

        for iv in range(len(ext_pas)):
            c = SkyCoord(ext_ra[iv], ext_dec[iv], unit="deg", frame="icrs")
            l = c.galactic.l.degree
            b = c.galactic.b.degree
            linelength_half = ext_scale * ext_Ps[iv]
            bs = [b - linelength_half * np.cos(ext_pas[iv]),
                  b + linelength_half * np.cos(ext_pas[iv])]
            ls = [l + linelength_half * np.sin(ext_pas[iv]),
                  l - linelength_half * np.sin(ext_pas[iv])]
            hp.projplot(ls, bs, c='cyan', lonlat=True, coord='G',
                        label='Panopoulou+2025' if iv == 0 else None)

    plt.legend(loc='upper right')
    plt.savefig('polarization_map.png', dpi=300, bbox_inches='tight')
    plt.show()


fits.info("diff_ebv_gnilc_lenz.fits")
eikona_fits = 'diff_ebv_gnilc_lenz.fits'

# ── my stars ────────────────────────────────────────────────────────────
ras_all, decs_all, Ps, angles = coordinates_os_stars()
angles = np.array(angles)
angles = np.radians(angles)
pas_forplot = angles + 0.   # coordang = 0

# ── external stars (loaded only when the toggle is on) ───────────────────────
ext_ra = ext_dec = ext_Ps_arr = ext_pas_forplot = None

if SHOW_EXTERNAL_CATALOG:
    ext_ra, ext_dec, ext_Ps_arr, ext_angles = coordinates_external_stars(EXTERNAL_CATALOG_PATH)
    ext_angles = np.array(ext_angles)
    ext_angles = np.radians(ext_angles)
    ext_pas_forplot = ext_angles + 0.   

segments_on_map(
    eikona_fits,
    ras_all, decs_all, Ps, pas_forplot,
    scale=0.5,
    ext_ra=ext_ra, ext_dec=ext_dec,
    ext_Ps=ext_Ps_arr, ext_pas=ext_pas_forplot,
    ext_scale=2.0,      # adjust if external segments look too long/short
)

plt.show()

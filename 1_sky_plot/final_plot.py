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
    c = SkyCoord(ra, dec, unit="deg", frame="icrs")  # defaults to ICRS frame
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
    c = SkyCoord(ra, dec, unit="deg", frame="icrs")  # defaults to ICRS frame
    targ_all_l.append(c.galactic.l.degree)
    targ_all_b.append(c.galactic.b.degree)
    # is it already observed or not?
    if not name in remaining_names:
        targ_done_l.append(c.galactic.l.degree)
        targ_done_b.append(c.galactic.b.degree)
fop.close()

f= plt.figure(figsize=(12,12))


hp.gnomview(diff_map, rot=[265, 80], min=-0.03, max=0.03, cmap='magma',
            xsize=150, ysize=150, fig=1,
            coord='G', reso=11, title="GnomView", unit="diff", format="%.2g")
hp.graticule()


hp.projscatter(targ_all_l, targ_all_b, marker='.', c='g', lonlat=True, coord='G', label='remaining')
hp.projscatter(targ_done_l, targ_done_b, marker='.', c='k', lonlat=True, coord='G', label='observed')
plt.legend()
plt.savefig('mark_plot.png')
plt.clf()



#changes i made: target
#targ_done_l=l
#targ_done_b=b

def appenzeller(l, b, evpa):
    '''Convert evpa to Galactic angle as in Appenzeller 1968 
        (almost, we put lNCP - l instead of l-lNCP, and add the angle instead of subtracting it)
        Input:
            - l: list/array of galactic longitudes in degrees
            - b: list/array of galactic latitudes in degrees
            - evpa: list/array of EVPA in radians in range [0,pi]
        Output:
            - array of angles with respect to NGP
    '''
    lp = 122.93200023
    bp = 27.12843
    ####should i do only for targ_done_l/b?
    targ_done_l = np.array(l) 
    targ_done_b = np.array(b)
#    l = np.array(l) 
#    b = np.array(b)
    evpa = np.array(evpa)
    Dt = np.arctan2(np.sin(np.radians(lp-targ_done_l)),\
        (np.cos(np.radians(targ_done_b))*np.tan(np.radians(bp))-\
            np.cos(np.radians(lp-targ_done_l))*np.sin(np.radians(targ_done_b))))
    return evpa + Dt


################################
#           second part - plotting polarization strips 
########################################################


class star:
    def __init__(self ,PD, PDerr, PA, PAerr):
        self.PD    = PD        # in %
        self.PDerr = PDerr
        self.PA    = PA        # in deg
        self.PAerr = PAerr
        
class Obs():
    def __init__(self,name,JD,q,qerr,u,uerr,x=0.0,y=0.0):
        self.name = name
        self.JD   = JD
        self.q    = ufloat(q,qerr)
        self.u    = ufloat(u,uerr)
        self.x    = x
        self.y    = y

    def getP(self):
        return umath.sqrt(self.q ** 2 + self.u ** 2)

    def getPA(self):
        PAval = 0.5*umath.atan2(self.u,self.q)
        PAunc = self.getSigma(self.getP().n,self.getP().s)
        return ufloat(np.degrees(PAval.n),np.degrees(PAunc))
    
    def correctInst(self,q_inst,u_inst):
        self.q = self.q - q_inst
        self.u = self.u - u_inst
    
    def correctRot(self,dPA):
        THETA = umath.radians(dPA)
        Qnew = self.q * umath.cos(2*THETA) - self.u * umath.sin(2*THETA)
        Unew = self.q * umath.sin(2*THETA) + self.u * umath.cos(2*THETA)
        self.q = Qnew
        self.u = Unew
        
    def EVPA_pdf(self,theta,P0):
        """
        EVPA measurements are also non-Gaussian and defined by the following
        probability density (Naghizadeh-Khouei & Clarke 1993):
        """
        g = 1/np.sqrt(np.pi)
        ita0 = float(P0)/np.sqrt(2) * np.cos(2 * theta)
        g = g * (g + ita0 * np.exp(ita0**2) * (1 + erf(ita0)))
        g = g * np.exp(-(float(P0)**2)/2)
        return g
    
    def int_eq(self,sigma,snr):
        """ This is the integral of EVPA probability density from -sigma to sigma """
        integ = integrate.quad(lambda x: self.EVPA_pdf(x,snr),-sigma,sigma)
        return abs(integ[0] - 0.68268949)
    
    def getSigma(self,pd,pd_err):
        snr = pd/pd_err
        if snr > 20:
            # it is a good approximation even for snr = 5
            return 0.5*1.0/float(snr)

        if snr < np.sqrt(2.0):
            pd = 0.0
        else:
            pd = np.sqrt(pd**2 - pd_err**2)

        snr = pd/pd_err

        res = minimize(self.int_eq, [np.pi/50], args=(snr,), method='Nelder-Mead', tol=1e-5) # np.pi/50 = 3.6 deg - just a reasonable guess
        if res.status != 0:
            print('Something is wrong with the EVPA uncertainty calculation:\n')
            return np.nan

        return res.x[0]

class Stand():
    def __init__(self, st):
        self.name = st
        stand = stand_pol.STANDARDS[self.name]
        self.PD    = ufloat(stand.PD/100., stand.PDerr/100.)
        self.PA    = ufloat(stand.PA, stand.PAerr)
        self.calcQU()
        
        
    def calcQU(self):
        self.q = self.PD * umath.cos( 2 * umath.radians(self.PA) )
        self.u = self.PD * umath.sin( 2 * umath.radians(self.PA) )
##################
##reading our 2 files and saving the needed information in one database


df_dat = pd.read_csv("Markkanen_final.dat", sep=r"\s+", engine="python")
df_dat.rename(columns=lambda x: x.lstrip('#'), inplace=True)  # remove '#' in "#Name"
#print(df_dat.head())

# Read the .csv file
df_csv = pd.read_csv("Markkanen_schedule_original.csv")
df_csv.rename(columns={"#name": "Name"}, inplace=True)
#print(df_csv.head())

##merge those 2 databases
df_csv_selected = df_csv[["Name", "ra", "dec"]]

df_merged = pd.merge(df_dat, df_csv_selected, on="Name", how="left")
#print(df_merged.head())
#save to CSV
df_merged.to_csv("merged_output.csv", index=False)

########################################################

def coordinates_os_stars():
    ra, dec, Ps, angles ,l_for_angle, b_for_angle= [], [], [], [] , [] , []

    for _, row in df_merged.iterrows():  # loop through each row
        x = row["ra"]          # RA from csv
        y = row["dec"]         # Dec from csv
        PD = row["P[%]"]       # P column from dat
        EVPA = row["PA[deg]"]  # PA column from dat
       
       
        ra.append(x)
        dec.append(y)
        Ps.append(PD)
        #angles.append(EVPA)
        c = SkyCoord(ra[-1], dec[-1], unit="deg", frame="icrs")  # defaults to ICRS frame
        l_for_angle.append(c.galactic.l.degree)
        b_for_angle.append(c.galactic.b.degree)
        np.set_printoptions(legacy='1.25')
        #print(l_for_angle[-1])
        evpa_corr= appenzeller(l_for_angle[-1], b_for_angle[-1], EVPA)
        angles.append(evpa_corr)
        # Debug print for each star
        #print(f"Star {row['Name']}: RA={x}, Dec={y}, P={PD}, PA={EVPA}")
        
    """ Convert angles from range [-pi,pi] or [-pi/2,pi/2]  to [0,pi). """
    for aa in range(len(angles)):
        if angles[aa] < 0:
            angles[aa] = 180 - abs(angles[aa])
        if angles[aa] == 180:
            angles[aa] = 0
        #angles[aa] = angles[aa]*180.0/np.pi
        #print(angles)
    return ra, dec, Ps, angles

stars = coordinates_os_stars()




def segments_on_map(fitsfile, ra, dec, Ps, pas, scale = 1000):
        '''
        Plot polarization segments on fits file.
        Input: 
        fitsfile: string, name of fits file
        ra: list of ra
        dec: list of dec
        pas: list of angles in radians. Angles are with respect to y axis of image.
             You need to rotate the real angle to have angles with respect to north.
        savename: string, name of plot to be saved
        coord: coordinate system of fits image
        scale: float/int, determines how long the segments will be in pixels
        '''
        # fig = aplpy.FITSFigure(fitsfile,figsize = (9,9))
        # fig.show_grayscale(invert=True)
        mapread = hp.read_map(fitsfile)
        fig = plt.figure(figsize=(12,12))
        hp.gnomview(mapread, rot=[265, 80], min=-0.03, max=0.03, cmap='magma',
                    xsize=150, ysize=150, fig=1,
                    coord='G', reso=11, title="GnomView", unit="diff", format="%.2g")
        hp.graticule()
        # fig.add_grid()
        # fig.grid._grid._linewidths = (0.4,)
        # fig.grid.show()
        
        # linelist1 = []
        # kukloi_ra, kukloi_dec, kukloi_polosi = [], [], []
        for iv in range(len(pas)):
            # xpix,ypix=fig.world2pixel(ra[iv],dec[iv])
            c = SkyCoord(ra[iv], dec[iv], unit="deg", frame="icrs")  # defaults to ICRS frame
            l = c.galactic.l.degree
            b = c.galactic.b.degree
            linelength_half= scale*Ps[iv]

            bs=[b-linelength_half*np.cos(pas[iv]),
               b+linelength_half*np.cos(pas[iv])]
            ls=[l+linelength_half*np.sin(pas[iv]),
               l-linelength_half*np.sin(pas[iv])]
            # x_world,y_world=fig.pixel2world([x[0],x[1]],[y[0],y[1]])
            # line=np.array([x_world,y_world])
            # linelist1.append(line)
            hp.projplot(ls, bs, c='g', lonlat=True, coord='G')
            
        # fig.show_lines(linelist1[:-1], layer='line', color='r', linewidths=1.3)
        # fig.show_lines([linelist1[-1]], layer='line1', color='cyan', linewidths=1.3)
        #fig.show_ellipses([35.3125], [25.3], 0.09, 0.07, angle=17, layer='ellipse', color='g') 
        plt.show()      

fits.info("diff_ebv_gnilc_lenz.fits")
eikona_fits = 'diff_ebv_gnilc_lenz.fits'

ras_all, decs_all, Ps, angles = coordinates_os_stars()  
angles = np.array(angles)
angles = np.radians(angles)
ra = ras_all
dec = decs_all
# Plot polarization segments on DSS 
coordang = 0.
pas_forplot = angles + coordang
   
segments_on_map( eikona_fits, ra, dec, Ps, pas_forplot, scale = 0.5 )

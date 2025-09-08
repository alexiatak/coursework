#!/usr/bin/env python3

'''
This script processes RoboPol polarization data

run as:
  python  proc_RBPL.py

'''

import os
import sys
import time
from uncertainties import ufloat
from uncertainties import umath
import math
import numpy as np
import scipy.optimize
from scipy import integrate
from scipy.special import erf
import random
import astropy.io.fits as pyfits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.aperture import aperture_photometry
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.visualization import simple_norm
import pickle



aperts     = [ 18.0,19.0,20.0,21.0,22.0] # radius in pixels
aperts_in  = [25.0,25.0,25.0,25.0,25.0]
aperts_out = [35.0,35.0,35.0,35.0,35.0]

MANUAL_FOR_ALL_SHOTS = True
#MANUAL_FOR_ALL_SHOTS = False

LEVELS = 99 # in percent change to 80 for bright stars

######## these are fixed - don't change them ########
GAIN = 2.687 # don't change this
BIAS = 562 # fixed value
#####################################################

class Shot():
    def __init__(self):
        self.fname      = None
        self.file_path  = None
        self.star_name  = None
        self.data       = None
        self.JD         = None
        self.CCDRON     = 8.14 # https://skinakas.physics.uoc.gr/en/images/infrastructure/gain2.jpg
        self.x0         = None
        self.x1         = None
        self.x2         = None
        self.x3         = None
        self.y0         = None
        self.y1         = None
        self.y2         = None
        self.y3         = None
        self.s0_flux     = [] # photons in the top spot
        self.s0_flux_err = [] # uncertainty of photons in the top spot
        self.s1_flux     = [] # photons in the bottom spot
        self.s1_flux_err = [] # uncertainty of right in the bottom spot
        self.s2_flux     = [] # photons in the right spot
        self.s2_flux_err = [] # uncertainty of photons in the right spot
        self.s3_flux     = [] # photons in the left spot
        self.s3_flux_err = [] # uncertainty of right in the left spot
        self.all_srcs   = None
        self.apertures  = []
        self.annuli     = []
        self.s0_bgr_arr  = []
        self.s1_bgr_arr  = []
        self.s2_bgr_arr  = []
        self.s3_bgr_arr  = []
        self.s0_bgr_median = []
        self.s1_bgr_median = []
        self.s2_bgr_median = []
        self.s3_bgr_median = []
        self.s0_bgr_std    = []
        self.s1_bgr_std    = []
        self.s2_bgr_std    = []
        self.s3_bgr_std    = []

    def loadImage(self):
        global BIAS
        hdulist = pyfits.open(self.file_path)
        self.data = hdulist[0].data - BIAS
        self.data[self.data < 0] = 0 # Replace values that became negative after subtracting bias
        header = hdulist[0].header
        self.star_name = hdulist[0].header['OBJECT']
        self.JD        = hdulist[0].header['JD']
        hdulist.close()

class ImageCoord:
    """ This class takes care of the mouse-clicks on top of the matplotlib canvas.
    The give x,y positions are recorded and used later to find centroids of the central
    target spots.
    """
    def __init__(self, xs, ys, point, fig):
        self.xs = xs
        self.ys = ys
        self.point = point
        self.fig = fig
        self.cid = point.figure.canvas.mpl_connect('button_press_event', self)

        ax = self.fig.gca()
        ax.set_title('Click on the star spots following the order given\nby numbers then close with middle mouse button click')
        ax.text(125.0, 200.0, "0", c='#FF00FF')
        ax.text(125.0,  45.0, "1", c='#FF00FF')
        ax.text(200.0, 125.0, "2", c='#FF00FF')
        ax.text( 45.0, 125.0, "3", c='#FF00FF')

    def __call__(self, event):
        if event.button == 1:
            self.xs.append(event.xdata - 1)
            self.ys.append(event.ydata - 1)
            ax = self.fig.gca()
            ax.scatter(self.xs,self.ys, c='#19FF19', s=100, zorder = 4)
            self.fig.canvas.draw()
            return
        elif event.button == 2:
            # got right click the current flare is finished
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close('all')
            return

        if event.inaxes!=self.point.axes: return

class Polar():
    def __init__(self):
        self.PD         = [] # fractional polarization
        self.PD_err     = []
        self.EVPA         = [] # EVPA
        self.EVPA_err     = []
        self.q            = []
        self.q_err        = []
        self.u            = []
        self.u_err        = []

class Obs():
    def __init__(self, q, u):
        self.q    = q
        self.u    = u

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

        res = scipy.optimize.minimize(self.int_eq, [np.pi/50], args=(snr,), method='Nelder-Mead', tol=1e-5) # np.pi/50 = 3.6 deg - just a reasonable guess
        if res.status != 0:
            print('Something is wrong with the EVPA uncertainty calculation:\n')
            return np.nan

        return res.x[0]

def CalcPol(shot):
    """
       Calculating Stokes Params, PD and PA from Fluxes
       considering left ray ordinary
    """
    p = Polar()
    for i in range(len(aperts)):
        F0 = ufloat(shot.s2_flux[i], shot.s2_flux_err[i])
        F1 = ufloat(shot.s3_flux[i], shot.s3_flux_err[i])
        F2 = ufloat(shot.s1_flux[i], shot.s1_flux_err[i])
        F3 = ufloat(shot.s0_flux[i], shot.s0_flux_err[i])

        Q = (F0-F1)/(F0+F1)
        U = (F2-F3)/(F2+F3)

        o    = Obs(Q, U)
        PD   = o.getP()
        EVPA = o.getPA()

        p.PD.append(PD.n)
        p.PD_err.append(PD.s)
        p.EVPA.append(EVPA.n)
        p.EVPA_err.append(EVPA.s)
        p.q.append(Q.n)
        p.q_err.append(Q.s)
        p.u.append(U.n)
        p.u_err.append(U.s)

    return p

def DetectStars(s):
    bgr_mean, bgr_median, bgr_std = sigma_clipped_stats(s.data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*bgr_std)
    sources = daofind(s.data - bgr_median)
    s.all_srcs = sources

def RunPhotom(s):
    # find the spots we need
    for i,src in enumerate(s.all_srcs):
        xcen = src['xcentroid']
        ycen = src['ycentroid']
        if math.sqrt((xcen-s.x0)**2+(ycen-s.y0)**2) < 4:
            # setting top spot more accurate
            s.x0 = xcen
            s.y0 = ycen
        elif math.sqrt((xcen-s.x1)**2+(ycen-s.y1)**2) < 4:
            # setting bottom spot more accurate
            s.x1 = xcen
            s.y1 = ycen
        elif math.sqrt((xcen-s.x2)**2+(ycen-s.y2)**2) < 4:
            # setting right spot more accurate
            s.x2 = xcen
            s.y2 = ycen
        elif math.sqrt((xcen-s.x3)**2+(ycen-s.y3)**2) < 4:
            # setting left spot more accurate
            s.x3 = xcen
            s.y3 = ycen

    # now measuring the spots
    positions = [(s.x0, s.y0), (s.x1, s.y1), (s.x2, s.y2), (s.x3, s.y3)]
    
    for i in range(len(aperts)):
        aperture = CircularAperture(positions, r=aperts[i])
        
        annulus_aperture = CircularAnnulus(positions, r_in=aperts_in[i], r_out=aperts_out[i])
        s.apertures.append(aperture) # there are both left and right here
        s.annuli.append(annulus_aperture)

        annulus_masks = annulus_aperture.to_mask(method='center')

        bkg_median = []
        
        # top spot
        annulus_data = annulus_masks[0].multiply(s.data)
        s0_bgr_arr = annulus_data[annulus_masks[0].data > 0]
        _, s0_bgr_median, s0_bgr_std = sigma_clipped_stats(s0_bgr_arr)
        bkg_median.append(s0_bgr_median)
        s.s0_bgr_arr.append(s0_bgr_arr)
        s.s0_bgr_median.append(s0_bgr_median)
        s.s0_bgr_std.append(s0_bgr_std)

        # bottom spot
        annulus_data = annulus_masks[1].multiply(s.data)
        s1_bgr_arr  = annulus_data[annulus_masks[1].data > 0]
        _, s1_bgr_median, s1_bgr_std = sigma_clipped_stats(s1_bgr_arr)
        bkg_median.append(s1_bgr_median)
        s.s1_bgr_arr.append(s1_bgr_arr)
        s.s1_bgr_median.append(s1_bgr_median)
        s.s1_bgr_std.append(s1_bgr_std)

        # right spot
        annulus_data = annulus_masks[2].multiply(s.data)
        s2_bgr_arr = annulus_data[annulus_masks[2].data > 0]
        _, s2_bgr_median, s2_bgr_std = sigma_clipped_stats(s2_bgr_arr)
        bkg_median.append(s2_bgr_median)
        s.s2_bgr_arr.append(s2_bgr_arr)
        s.s2_bgr_median.append(s2_bgr_median)
        s.s2_bgr_std.append(s2_bgr_std)

        # left spot
        annulus_data = annulus_masks[3].multiply(s.data)
        s3_bgr_arr = annulus_data[annulus_masks[3].data > 0]
        _, s3_bgr_median, s3_bgr_std = sigma_clipped_stats(s3_bgr_arr)
        bkg_median.append(s3_bgr_median)
        s.s3_bgr_arr.append(s3_bgr_arr)
        s.s3_bgr_median.append(s3_bgr_median)
        s.s3_bgr_std.append(s3_bgr_std)

        bkg_median = np.array(bkg_median)
        phot = aperture_photometry(s.data, aperture)
        phot['annulus_median'] = bkg_median
        phot['aper_bkg'] = bkg_median * aperture.area
        phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
        
        # Signal
        s.s0_flux.append(phot[0]['aper_sum_bkgsub'] * GAIN)
        s.s1_flux.append(phot[1]['aper_sum_bkgsub'] * GAIN)
        s.s2_flux.append(phot[2]['aper_sum_bkgsub'] * GAIN)
        s.s3_flux.append(phot[3]['aper_sum_bkgsub'] * GAIN)

        # Noise
        sN0 = math.sqrt( phot[0]['aper_sum_bkgsub'] * GAIN + aperture.area * (1 + aperture.area/annulus_aperture.area) * ( phot[0]['annulus_median'] * GAIN +  s.CCDRON**2) )
        sN1 = math.sqrt( phot[1]['aper_sum_bkgsub'] * GAIN + aperture.area * (1 + aperture.area/annulus_aperture.area) * ( phot[1]['annulus_median'] * GAIN +  s.CCDRON**2) )
        sN2 = math.sqrt( phot[2]['aper_sum_bkgsub'] * GAIN + aperture.area * (1 + aperture.area/annulus_aperture.area) * ( phot[2]['annulus_median'] * GAIN +  s.CCDRON**2) )
        sN3 = math.sqrt( phot[3]['aper_sum_bkgsub'] * GAIN + aperture.area * (1 + aperture.area/annulus_aperture.area) * ( phot[3]['annulus_median'] * GAIN +  s.CCDRON**2) )
        
        s.s0_flux_err.append(sN0)
        s.s1_flux_err.append(sN1)
        s.s2_flux_err.append(sN2)
        s.s3_flux_err.append(sN3)
    return

def MakeReg():
    fop = open(os.path.join(os.getcwd(),'ds9.reg'),'w')
    fop.write('physical\n')
    for spot in spots:
        fop.write('circle('+str(spot[0])+','+str(spot[1])+',15.1)\n')
    fop.close()

def Plot(p, s):
    # prepare folder for figures
    CWD = os.getcwd()
    img_folder = os.path.join(CWD, s.fname)
    os.makedirs(img_folder, exist_ok=True)


    #  apertures
    for i in range(len(aperts)):
        norm = simple_norm(s.data, 'sqrt', percent=LEVELS)
        plt.imshow(s.data, norm=norm)
        s.apertures[i].plot(color='white', lw=0.5)
        s.annuli[i].plot(color='red', lw=0.5)
        xs = [s.x0, s.x1, s.x2, s.x3]
        ys = [s.y0, s.y1, s.y2, s.y3]
        plt.xlim(min(xs) - 60, max(xs) + 60)
        plt.ylim(min(ys) - 60, max(ys) + 60)
        #plt.axes().set_aspect('equal', 'datalim')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(img_folder, s.fname+'_'+str(i+1)+'_'+'.png'),bbox_inches='tight')
        plt.close('all')
        plt.cla()
        plt.clf()


    #  bckgr historgrams
    figure = plt.figure(figsize=(6, 7.5), dpi=150)
    figure.subplots_adjust(hspace=0.18)
        
    fig0 = plt.subplot(411)
    # top
    fig0_x_min = s.s0_bgr_median[0] - 5 * s.s0_bgr_std[0]
    fig0_x_max = s.s0_bgr_median[0] + 5 * s.s0_bgr_std[0]
    plt.hist(s.s0_bgr_arr[0], density=True, bins=30, range=[fig0_x_min, fig0_x_max], alpha = 0.9, color='#f0b27a')
    plt.axvline(s.s0_bgr_median[0], color='#5dade2')
        
    fig1 = plt.subplot(412)
    # bottom
    fig1_x_min = s.s1_bgr_median[0] - 5 * s.s1_bgr_std[0]
    fig1_x_max = s.s1_bgr_median[0] + 5 * s.s1_bgr_std[0]
    plt.hist(s.s1_bgr_arr[0], density=True, bins=30, range=[fig1_x_min, fig1_x_max], alpha = 0.9, color='#f0b27a')
    plt.axvline(s.s1_bgr_median[0], color='#5dade2')

    fig2 = plt.subplot(413)
    # right
    fig2_x_min = s.s2_bgr_median[0] - 5 * s.s2_bgr_std[0]
    fig2_x_max = s.s2_bgr_median[0] + 5 * s.s2_bgr_std[0]
    plt.hist(s.s2_bgr_arr[0], density=True, bins=30, range=[fig2_x_min, fig2_x_max], alpha = 0.9, color='#f0b27a')
    plt.axvline(s.s2_bgr_median[0], color='#5dade2')

    fig3 = plt.subplot(414)
    # left
    fig3_x_min = s.s3_bgr_median[0] - 5 * s.s3_bgr_std[0]
    fig3_x_max = s.s3_bgr_median[0] + 5 * s.s3_bgr_std[0]
    plt.hist(s.s3_bgr_arr[0], density=True, bins=30, range=[fig3_x_min, fig3_x_max], alpha = 0.9, color='#f0b27a')
    plt.axvline(s.s3_bgr_median[0], color='#5dade2')

    plt.savefig(os.path.join(img_folder, s.fname+'_bgr_'+str(i+1)+'_'+'.png'),bbox_inches='tight')
    plt.close('all')
    plt.cla()
    plt.clf()

    #  polar params vs aperture
    figure = plt.figure(figsize=(6, 6), dpi=150)
    figure.subplots_adjust(hspace=0.1)
    fig1 = plt.subplot(211)
    plt.errorbar(aperts,p.PD,xerr=0,yerr=p.PD_err,color='k', marker='o', markeredgecolor='k', ms=8., capsize=0, linestyle='None', fmt='')
    fig1.set_xticklabels([])
    plt.ylabel('PD [frac]')
    fig1 = plt.subplot(212)
    plt.errorbar(aperts,p.EVPA,xerr=0,yerr=p.EVPA_err,color='k', marker='o', markeredgecolor='k', ms=8., capsize=0, linestyle='None', fmt='')
    plt.xlabel('Aperture radius [pix]')
    plt.ylabel('EVPA [deg]')

    plt.savefig(os.path.join(img_folder, 'PD_apert.png'),bbox_inches='tight')
    plt.close('all')
    plt.cla()
    plt.clf()


def Output(p, s):
    # prepare folder for files
    CWD = os.getcwd()
    out_folder = os.path.join(CWD, s.fname)
    os.makedirs(out_folder, exist_ok=True)

    fop = open(os.path.join(out_folder, s.star_name + '_result.dat'),'w')
    fop.write('# {:s}\n'.format(s.star_name))
    fop.write('# {:12.5f}\n'.format(s.JD))
    fop.write('# apert     q         sq           u           su         PD          sPD        EVPA        sEVPA       x0     y0    x1       y1      x2     y2     x3      y3\n')
    for i,ap in enumerate(aperts):
        ostr = '{:4.1f}'.format(ap) + '{:12.5f}'.format(p.q[i]) + '{:12.5f}'.format(p.q_err[i])
        ostr += '{:12.5f}'.format(p.u[i]) + '{:12.5f}'.format(p.u_err[i])
        ostr += '{:12.5f}'.format(p.PD[i]) + '{:12.5f}'.format(p.PD_err[i])
        ostr += '{:12.5f}'.format(p.EVPA[i]) + '{:12.5f}'.format(p.EVPA_err[i])
        ostr += '{:8.2f}'.format(s.x0) + '{:8.2f}'.format(s.y0)
        ostr += '{:8.2f}'.format(s.x1) + '{:8.2f}'.format(s.y1)
        ostr += '{:8.2f}'.format(s.x2) + '{:8.2f}'.format(s.y2)
        ostr += '{:8.2f}'.format(s.x3) + '{:8.2f}'.format(s.y3)
        fop.write(ostr+'\n')
    fop.close()

    # saving the entire s instance
    with open(os.path.join(out_folder, s.star_name + '_result.pickle'), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump([p,s], f, pickle.HIGHEST_PROTOCOL)


def getFileList():
    """ Scans subfolder starting with 24 and looks for .fits files inside
        returns this list with the full paths
    """
    CWD = os.getcwd()
    subfolders = list( filter( lambda x: x.startswith("24"), os.listdir(CWD) ) )

    if len(subfolders) < 1:
        print("There is no subfolder starting with 24 that contains raw fits data")
        exit(1)

    raw_data_subfolder = os.path.join(CWD, subfolders[0])

    fits_names = list( filter( lambda x: x.endswith(".fits"), os.listdir(raw_data_subfolder) ) )
    fits_files = list( map(lambda x: os.path.join(raw_data_subfolder, x), fits_names) )
    return fits_files

def GetSpotsCoords(shot):
    """ Rises a window to get users input and set initial guess for x,y of target spots
    """
    # using matplotlib canvas
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    font = {'size'   : 20, 'family' : 'sans-serif'}
    mpl.rc('font', **font)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    x1 = 900
    x2 = 1160
    y1 = 910
    y2 = 1170

    image_mask = shot.data[y1:y2,x1:x2]
    lmin = np.median(image_mask) - 3*np.std(image_mask)
    lmax = np.median(image_mask) + 3*np.std(image_mask)
    levels = np.linspace(lmin,lmax,31)
    ax.contourf(image_mask,levels=levels, cmap = plt.cm.gray)

    xs = []
    ys = []

    point = ax.scatter([],[], c='#19FF19', s=100)
    ImageCoord(xs, ys, point, fig)
    plt.show(block=True)

    xs = list( map(lambda x: x + x1, xs) )
    ys = list( map(lambda x: x + y1, ys) )
    rs = [5.0, 5.0, 5.0, 5.0]

    plt.clf()
    plt.cla()
    plt.close(fig)
    return xs, ys

def ProcObj():
    
    # iterate through all 4 files to get counts
    shots = getFileList()
    shots.sort()

    for i, fits_path in enumerate(shots):
        s = Shot()
        s.fname = os.path.basename(fits_path)
        s.file_path = fits_path
        s.loadImage()

        if (i==0) or MANUAL_FOR_ALL_SHOTS:
            xs_0, ys_0 = GetSpotsCoords(s)

        s.x0, s.x1, s.x2, s.x3 = xs_0
        s.y0, s.y1, s.y2, s.y3 = ys_0

        DetectStars(s)
        RunPhotom(s)
        p = CalcPol(s)
        Plot(p, s)
        Output(p, s)

        with open('apertures.txt', 'a') as file_out:
            file_out.write(s.fname + '   ' + '0.0\n')

if __name__ == "__main__":
    with open('apertures.txt', 'w') as file_out:
            file_out.write('#Filename  optimal_apeture\n')
    ProcObj()







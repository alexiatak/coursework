
import numpy as np
from uncertainties import ufloat
from uncertainties import umath

from scipy import optimize, interpolate
from scipy.special import erf
from scipy.optimize import minimize
import scipy.integrate as integrate

class star:
    def __init__(self, PD=None, PDerr=None, PA=None, PAerr=None, Q=None, Qerr=None, U=None, Uerr=None):
        if (PD is not None) and (PA is not None):
            self.PD    = PD        # in %
            self.PDerr = PDerr
            self.PA    = PA        # in deg
            self.PAerr = PAerr
        
            self.P     = ufloat(self.PD/100., self.PDerr/100.)
            self.EVPA  = ufloat(umath.radians(self.PA), umath.radians(self.PAerr))
            self.q     = None      # fractions
            self.u     = None
            self.calcQU()
            self.Q     = self.q.n      # fractions
            self.Qerr  = self.q.s
            self.U     = self.u.n
            self.Uerr  = self.u.s
        elif (Q is not None) and (U is not None):
            self.Q     = Q      # fractions
            self.Qerr  = Qerr
            self.U     = U
            self.Uerr  = Uerr
            self.q     = ufloat(self.Q, self.Qerr)
            self.u     = ufloat(self.U, self.Uerr)
            self.PD    = self.getP().n * 100
            self.PDerr = self.getP().s * 100
            self.P     = self.getP()
            self.PA    = np.degrees(self.getPA().n)              # in deg
            self.PAerr = np.degrees(self.getPA().s)
            self.EVPA  = self.getPA()

    def calcQU(self):
        self.q = self.P * umath.cos(2*self.EVPA)
        self.u = self.P * umath.sin(2*self.EVPA)
        
    def getP(self):
        return umath.sqrt(self.q ** 2 + self.u ** 2)
    
    def getPA(self):
        PAval = 0.5*umath.atan2(self.u,self.q)
        PAunc = self.getSigma(self.getP().n,self.getP().s)
        return ufloat(PAval.n,PAunc)
    
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

STANDARDS = {# polarized
             "HD183143"   : star(PD=5.995,PDerr=0.032,PA=179.4,PAerr=0.15),      # averaged B and V from https://articles.adsabs.harvard.edu/pdf/1982ApJ...262..732H
             # unpolarized
             "BD+28.4211" : star(PD=0.054,PDerr=0.016,PA=41.0,PAerr=9.0),   # averaged B and V from https://articles.adsabs.harvard.edu/pdf/1982ApJ...262..732H
             "BD+32.3739" : star(PD=0.024,PDerr=0.009,PA=63.0,PAerr=13.0),   # averaged B and V from https://articles.adsabs.harvard.edu/pdf/1982ApJ...262..732H
             "BD+33.2642" : star(PD=0.18,PDerr=0.015,PA=19.2,PAerr=2.4),     # averaged B and V from https://articles.adsabs.harvard.edu/pdf/1982ApJ...262..732H but may have changed Skalidis18
             "BD+40.2704" : star(PD=0.07,PDerr=0.02,PA=57,PAerr=9),          # unkn band Berdyugin2002
             "HD154892"   : star(PD=0.05,PDerr=0.03,PA=0.0,PAerr=0.0),       # B band from Turnishek90
             }



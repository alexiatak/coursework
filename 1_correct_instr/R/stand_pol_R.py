
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
             "BD+57.2615" : star(PD=2.02,PDerr=0.05,PA=41.00,PAerr=1.0),     # Whittet92
             "BD+59.389"  : star(PD=6.43,PDerr=0.022,PA=98.14,PAerr=0.1),    # Schmidt92
             "BD+64.106"  : star(PD=5.15,PDerr=0.098,PA=96.74,PAerr=0.54),   # Schmidt92
             "CMaR1_24"   : star(PD=3.18,PDerr=0.09,PA=86.00,PAerr=1.0),     # Whittet92
             "CygOB2_14"  : star(PD=3.13,PDerr=0.05,PA=86.00,PAerr=1.0),     # Whittet92
             "HD147283"   : star(PD=1.59,PDerr=0.03,PA=174.0,PAerr=1.0),     # Whittet92
             "HD147343"   : star(PD=0.43,PDerr=0.05,PA=151.0,PAerr=3.0),     # Whittet92
             "HD150193"   : star(PD=5.19,PDerr=0.05,PA=56.00,PAerr=1.0),     # Whittet92
             "HD154445"   : star(PD=3.63,PDerr=0.01,PA=90.0,PAerr=0.1),      # Hsu82
             "HD155197"   : star(PD=4.274,PDerr=0.027,PA=102.88,PAerr=0.18), # Schmidt92
             "HD161056"   : star(PD=4.012,PDerr=0.032,PA=67.33,PAerr=0.23),  # Schmidt92
             "HD183143"   : star(PD=5.9,PDerr=0.05,PA=179.2,PAerr=0.2),      # Hsu82
             "HD204827"   : star(PD=4.893,PDerr=0.029,PA=59.1,PAerr=0.17),   # Schmidt92
             "HD215806"   : star(PD=1.83,PDerr=0.04,PA=66.0,PAerr=1.0),      # Whittet92
             "HD236633"   : star(PD=5.376,PDerr=0.028,PA=93.04,PAerr=0.15),  # Schmidt92
             "Hiltner960" : star(PD=5.210,PDerr=0.029,PA=54.54,PAerr=0.16),  # Schmidt92
             "VICyg12"    : star(PD=7.893,PDerr=0.037,PA=116.23,PAerr=0.14),# Schmidt92
             # unpolarized
             "BD+28.4211" : star(PD=0.054,PDerr=0.027,PA=54.22,PAerr=0.0),   # Schmidt92
             "BD+32.3739" : star(PD=0.025,PDerr=0.017,PA=35.79,PAerr=0.0),   # Schmidt92
             "BD+33.2642" : star(PD=0.20,PDerr=0.15,PA=78.0,PAerr=20.0),     # Skalidis18
             "BD+40.2704" : star(PD=0.07,PDerr=0.02,PA=57,PAerr=9),          # Berdyugin2002
             "G191B2B"    : star(PD=0.061,PDerr=0.038,PA=147.65,PAerr=0.0),  # Schmidt92
             "HD14069"    : star(PD=0.022,PDerr=0.019,PA=156.57,PAerr=0.0),  # Schmidt92
             "HD154892"   : star(PD=0.05,PDerr=0.03,PA=0.0,PAerr=0.0),       # Turnishek90
             "HD212311"   : star(PD=0.034,PDerr=0.021,PA=50.99,PAerr=0.0),   # Schmidt92
             "HD21447"    : star(PD=0.051,PDerr=0.020,PA=171.49,PAerr=0.0),  # Schmidt92
             "HD94851"    : star(PD=0.057,PDerr=0.018,PA=0.0,PAerr=0.0),     # Turnishek90
             #"WD2149+021" : star(0.141,0.139,175.93,18.49),
             "WD2149+021" : star(PD=0.05,PDerr=0.006,PA=117.36,PAerr=3.42),  # Cikota17
             "ST_B_1725+1152_11"  : star(Q=-0.0080,Qerr=0.0017,U=0.0050,Uerr=0.0009),  # Blinov23
             "ST_B_1725+1152_35"  : star(Q=-0.0044,Qerr=0.0018,U=0.0052,Uerr=0.0011),  # Blinov23
             "ST_B_1959+6508_38"  : star(Q=-0.0033,Qerr=0.0012,U=-0.0155,Uerr=0.0008),  # Blinov23
             "ST_B_1959+6508_104" : star(Q=-0.0005,Qerr=0.0018,U=-0.0095,Uerr=0.0012),  # Blinov23
             "ST_B_2022+7611_1" : star(Q=0.0062,Qerr=0.0010,U=-0.0021,Uerr=0.0010),  # Blinov23
             "ST_B_2042+7508_28" : star(Q=0.0037,Qerr=0.0010,U=0.0102,Uerr=0.0011),  # Blinov23
             "ST_B_2253+1608_23" : star(Q=-0.0003,Qerr=0.0010,U=-0.0022,Uerr=0.0011),  # Blinov23
             "ST_B_2340+8015_34" : star(Q=0.0030,Qerr=0.0022,U=0.0119,Uerr=0.0016),  # Blinov23
             "ST_L_93_317" : star(Q=0.0011,Qerr=0.0009,U=-0.0020,Uerr=0.0009),  # Blinov23
             "ST_L_93_424" : star(Q=0.0008,Qerr=0.0010,U=-0.0024,Uerr=0.0011),  # Blinov23
             "ST_L_94_251" : star(Q=0.0039,Qerr=0.0010,U=0.0017,Uerr=0.0010),  # Blinov23
             "ST_L_104_334" : star(Q=-0.0001,Qerr=0.0015,U=0.0019,Uerr=0.0016),  # Blinov23
             "ST_L_106_700" : star(Q=-0.0040,Qerr=0.0009,U=0.0029,Uerr=0.0007),  # Blinov23
             "ST_L_107_599" : star(Q=-0.0067,Qerr=0.0011,U=0.0078,Uerr=0.0010),  # Blinov23
             'ST_L_109_381' : star(Q=-0.0139,Qerr=0.0013,U=0.0047,Uerr=0.0009),  # Blinov23
             'ST_L_110_233' : star(Q=0.0240,Qerr=0.0017,U=0.0106,Uerr=0.0007),  # Blinov23
             "ST_L_111_1965" : star(Q=-0.0104,Qerr=0.0009,U=0.0064,Uerr=0.0010),  # Blinov23
             "ST_L_112_822" : star(Q=-0.0033,Qerr=0.0015,U=0.0010,Uerr=0.0012),  # Blinov23
             "ST_L_115_420" : star(Q=-0.0017,Qerr=0.0010,U=-0.0011,Uerr=0.0010),  # Blinov23
             "ST_L_PG1633+099D" : star(Q=-0.0036,Qerr=0.0020,U=0.0038,Uerr=0.0013),  # Blinov23
             "ST_Z_BD+18.2549" : star(Q=-0.0002,Qerr=0.0011,U=0.0017,Uerr=0.0009),  # Blinov23
             "ST_Z_BD+30.2431" : star(Q=-0.0005,Qerr=0.0013,U=0.0020,Uerr=0.0010),  # Blinov23
             "ST_Z_BD+35.2256" : star(Q=0.0006,Qerr=0.0023,U=0.0013,Uerr=0.0009),  # Blinov23
             "ST_Z_HD85471" : star(Q=0.0005,Qerr=0.0013,U=0.0013,Uerr=0.0014),  # Blinov23
             "ST_Z_HD87582" : star(Q=0.0011,Qerr=0.0013,U=0.0010,Uerr=0.0009),  # Blinov23
             }


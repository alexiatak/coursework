
import pickle
from scipy import optimize, interpolate
from scipy.special import erf
from scipy.optimize import minimize
import scipy.integrate as integrate
from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties import umath
import numpy as np

INP_FILE = '../collect_filt_raw_data/Markkanen/Markkanen.dat'
OUT_FILE = 'Markkanen_inst_corr_separ_shots.dat'

Deattachments = [2460310.5,2460676.5]#,2461041.5]
years = ['2024']#,'2025']

class Obs():
    def __init__(self,name,JD,q,qerr,u,uerr,x,y):
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
        self.PA = ufloat(PAval.n,PAunc)
        return umath.degrees(self.PA)
    
    def getPADiffCat(self, PA_cat, PA_cat_err):
        PAcat = ufloat(PA_cat, PA_cat_err)
        self.PAdiff = umath.degrees(self.PA) - PAcat
        if self.PAdiff.n < -90:
            self.PAdiff = ufloat(self.PAdiff.n + 180, self.PAdiff.s)
        elif self.PAdiff.n > 90:
            self.PAdiff = ufloat(self.PAdiff.n - 180, self.PAdiff.s)
    
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

        res = minimize(self.int_eq, [np.pi/50], args=(snr,), method='Nelder-Mead', tol=1e-4) # np.pi/50 = 3.6 deg - just a reasonable guess
        if res.status != 0:
            print('Something is wrong with the EVPA uncertainty calculation:\n')
            return np.nan

        return res.x[0]

def readInstPol():
    instQ = {}
    instU = {}
    for spn in range(len(years)):
        year = years[spn]
        with open('../create_model/fit_Q'+year+'.pickle', 'rb') as f:
            instQ[year] = pickle.load(f)

        with open('../create_model/fit_U'+year+'.pickle', 'rb') as f:
            instU[year] = pickle.load(f)

    return instQ, instU

def readEVPAZP():
    fop = open('../create_model/EVPA_zp.txt')
    fop.readline()
    line = fop.readline()
    sl = line.rstrip('\n').split()
    fop.close()
    EVPA_ZP = ufloat( float(sl[0]), float(sl[1]) )
    return EVPA_ZP

def readMarkData():
    """Reading raw Markkanen data
        #Name   JD    P   sP    PA[deg]  sPA[deg]   q        sq       u        su   x0    y0    x1   y1   x2    y2    x3    y3
    """
    data = []
    fop = open(INP_FILE)
    line = fop.readline()

    for line in fop.readlines():
        sl = line.split()
        star_name = sl[0]
        JD    = float(sl[1])
        q     = float(sl[6])
        q_err = float(sl[7])
        u     = float(sl[8])
        u_err = float(sl[9])

        x0 = float(sl[10])
        x1 = float(sl[12])
        x = (x0 + x1)/2.
        y2 = float(sl[15])
        y3 = float(sl[17])
        y = (y2 + y3)/2.

        s = Obs(star_name, JD, q, q_err, u, u_err, x, y)
        data.append(s)

    fop.close()
    return data

def func(x, y, a, b, c, d, e, f):
    return a*(x**2) + b*(y**2) + c*(x*y) + d*x + e*y + f

def correctInstPol(data, instQ, instU, EVPA_ZP):
    # correcting data for the x,y & time dependent instrumental polarization
    corr_data = []
    for o in data:
        year = 'NONE' # for safety
        for i in range(len(years)):
            JD_LL = Deattachments[i]
            JD_UL = Deattachments[i+1]
            if JD_LL < o.JD < JD_UL:
                year = years[i]

        QmodelPrediction = func(o.x, o.y, *instQ[year][0]) # the second element in instQ, instU
        UmodelPrediction = func(o.x, o.y, *instU[year][0]) # is covariance we don't need it
        o.correctInst(QmodelPrediction, UmodelPrediction)
        o.correctRot(-EVPA_ZP)
        corr_data.append(o)
    return corr_data

def writeDataSeparateShots(corr_data):
    fout = open(OUT_FILE, 'w')
    fout.write("# Star   JD           PD[%]   sPD  EPVA[deg] sEVPA     q       sq        u       su\n")
    for o in corr_data:
        out_str  = '{name: <16}'.format(name=o.name)
        out_str += "{:13.5f}".format(o.JD) + "  "
        out_str += "{:7.2f}".format(round(o.getP().n*100, 2))
        out_str += "{:7.2f}".format(round(o.getP().s*100, 2))
        out_str += "{:8.2f}".format(round(o.getPA().n, 2))
        out_str += "{:8.2f}".format(round(o.getPA().s, 2))
        out_str += "{:9.5f}".format(round(o.q.n, 5))
        out_str += "{:9.5f}".format(round(o.q.s, 5))
        out_str += "{:9.5f}".format(round(o.u.n, 5))
        out_str += "{:9.5f}".format(round(o.u.s, 5))
        out_str += "\n"
        fout.write(out_str)
    
    fout.close()

def combine_measurements(star_name, corr_data):
    """ Combines multiple measurements of the same star into weighted mean
        and plots the q - u plane to control outliers
    """
    data = []
    for o in corr_data:
        if o.name == star_name:
            data.append(o)

    # calulating weighted average


    # plotting
    plt.rc('text', usetex=True)
    figure = plt.figure(figsize = (6, 6.0), dpi = 150)
    figure.subplots_adjust(hspace = 0.1)
    fig1 = plt.subplot(111)

    plt.xlabel("q")
    plt.ylabel("u")
    plt.errorbar(s1.q.n,s1.u.n,xerr=s1.q.s,yerr=s1.u.s,c='b',label=s1.name)
    plt.errorbar(s2.q.n,s2.u.n,xerr=s2.q.s,yerr=s2.u.s,c='b')
    plt.errorbar(s3.q.n,s3.u.n,xerr=s3.q.s,yerr=s3.u.s,c='b')
    plt.errorbar(s4.q.n,s4.u.n,xerr=s4.q.s,yerr=s4.u.s,c='b')
    plt.errorbar(s5.q.n,s5.u.n,xerr=s5.q.s,yerr=s5.u.s,c='b')

    # the centroid value
    plt.errorbar(q_z.n,u_z.n,xerr=q_z.s,yerr=u_z.s,c='yellow',label='median')

    plt.legend()

    plt.axvline(0)
    plt.axhline(0)

    return comb_star

def writeData(comb_star):
    return

if __name__ == "__main__":
    print("#"*80)
    print("Working on 2024 Markkanen cloud data")
    print("#"*80)
    # reading instrumental Stokes parameters and the EVPA zero-point
    instQ, instU = readInstPol()
    EVPA_ZP = readEVPAZP()
    
    # reading data to correct
    data = readMarkData()
    
    # correcting data
    corr_data = correctInstPol(data, instQ, instU, EVPA_ZP)
    
    # output corrected data
    writeDataSeparateShots(corr_data)

    uniq_star_names = list( map(lambda x: x.name, corr_data ) )
    for star_name in uniq_star_names:
        comb_star = combine_measurements(star_name, corr_data)
        writeData(comb_star)


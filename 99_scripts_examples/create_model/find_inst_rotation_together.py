#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./MNRAS_Style.mplstyle')
import os
import matplotlib as mpl
import math
from astropy import stats
from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties import umath
from statsmodels.stats.weightstats import DescrStatsW
import stand_pol
from scipy import optimize, interpolate
from scipy.special import erf
from scipy.optimize import minimize
import scipy.integrate as integrate
import pickle

class Star():
    def __init__(self, name):
        self.name = name
        self.variability = None
        self.wmPD = None
        self.stdPD = None
        self.wmPA = None
        self.stdPA = None
        self.wmQ = None
        self.stdQ = None
        self.wmU = None
        self.stdU = None
        self.lenJD = None
        self.PAdiff_val = None
        self.PAdiff_unc = None

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
        return self.PA
    
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

def get_wm(data, st):
    JDs = list( map(lambda x: x.JD - 2450000.5, data) )
    
    Qs = list( map(lambda x: x.q.n, data) )
    Qerrs = list( map(lambda x: x.q.s, data) )
    
    Us = list( map(lambda x: x.u.n, data) )
    Uerrs = list( map(lambda x: x.u.s, data) )

    if len(Qerrs) > 1:    
        weighted_stats = DescrStatsW(Qs, weights=list(map(lambda x:1.0/x,Qerrs)), ddof=0)
        wmQ = weighted_stats.mean
        stdQ = weighted_stats.std
        std_meanQ = weighted_stats.std_mean
    
        weighted_stats = DescrStatsW(Us, weights=list(map(lambda x:1.0/x,Uerrs)), ddof=0)
        wmU = weighted_stats.mean
        stdU = weighted_stats.std
        std_meanU = weighted_stats.std_mean
    elif len(Qerrs) == 1:
        wmQ = Qs[0]
        stdQ = Qerrs[0]
        std_meanQ = Qerrs[0]
        
        wmU = Us[0]
        stdU = Uerrs[0]
        std_meanU = Uerrs[0]
    else:
        wmQ = float('nan')
        stdQ = float('nan')
        std_meanQ = float('nan')
        
        wmU = float('nan')
        stdU = float('nan')
        std_meanU = float('nan')
    
    o = Obs("Name",0.0,wmQ,std_meanQ,wmU,std_meanU,0,0)

    P = o.getP() * 100
    PA = umath.degrees(o.getPA())
    st.wmPD, st.stdPD = P.n, P.s
    st.wmPA, st.stdPA = PA.n, PA.s
    st.wmQ,  st.stdQ  = wmQ, std_meanQ
    st.wmU,  st.stdU  = wmU, std_meanU
    
    # calculate difference with the catalogued value
    stand = stand_pol.STANDARDS[st.name]
    PA_rbpl = umath.radians( ufloat(st.wmPA,st.stdPA) )
    PA_cat  = umath.radians( ufloat(stand.PA,stand.PAerr) )
    PAdiff =  umath.degrees( umath.atan2(umath.sin(PA_rbpl-PA_cat), umath.cos(PA_rbpl-PA_cat)) )
    
    if PAdiff.n > 90:
        st.PAdiff_val = PAdiff.n - 180
    elif PAdiff.n < -90:
        st.PAdiff_val = PAdiff.n + 180
    else:
        st.PAdiff_val = PAdiff.n

    st.PAdiff_unc = PAdiff.s
    return st

def ReadData(band, star):
    data = []
    file_path = os.path.join('../collect_filt_raw_data/standards', 'standards_raw.dat')
    if os.path.exists(file_path):
        fop = open(file_path)
        fop.readline()
        for line in fop.readlines():
            if line.startswith("#"):
                continue

            sl = line.split()
            name = sl[0]
            if name == star:
                JD = float(sl[1])
                q = float(sl[6])
                qerr = float(sl[7])
                u = float(sl[8])
                uerr = float(sl[9])

                x0 = float(sl[10])
                x1 = float(sl[12])
                x = (x0 + x1)/2.
                y2 = float(sl[15])
                y3 = float(sl[17])
                y = (y2 + y3)/2.

                o = Obs(star, JD, q, qerr, u, uerr, x, y)
                data.append(o)
        fop.close()

    return data

def func(x, y, a, b, c, d, e, f):
    return a*(x**2) + b*(y**2) + c*(x*y) + d*x + e*y + f

def CorrInstPol(data):
    # correcting data for the x,y & time dependent instrumental polarization
    corr_data_R = []
    # in Sep 2022 one of the lenses in the collimator had been deattached then it had been fixed later
    # therefore we split 2022 into two pieces
    Deattachments = [2460310.5,2460676.5]#,2461041.5]
    years = ['2024']#,'2025']

    fittedParametersQ = {}
    fittedParametersU = {}
    for spn in range(len(years)):
        year = years[spn]
        with open('./fit_Q'+year+'.pickle', 'rb') as f:
            fittedParametersQ[year] = pickle.load(f)

        with open('./fit_U'+year+'.pickle', 'rb') as f:
            fittedParametersU[year] = pickle.load(f)

    for o in data:
        year = 'NONE' # for safety
        for i in range(len(years)):
            JD_LL = Deattachments[i]
            JD_UL = Deattachments[i+1]
            if JD_LL < o.JD < JD_UL:
                year = years[i]

        QmodelPrediction = func(o.x, o.y, *fittedParametersQ[year][0]) # the second element is covariance
        UmodelPrediction = func(o.x, o.y, *fittedParametersU[year][0])
        o.correctInst(QmodelPrediction, UmodelPrediction)
        corr_data_R.append(o)

    return corr_data_R

def plot(wa_dPA_mean, wa_dPA_std, stars_data):
    PAdiff_vals = []
    PAdiff_unc  = []
    Names = []
    for st in stars_data:
        if np.isnan(st.PAdiff_val) or np.isnan(st.PAdiff_unc):
            continue
        Names.append(st.name)
        PAdiff_vals.append(st.PAdiff_val)
        PAdiff_unc.append(st.PAdiff_unc)
    
    #font = {'size'   : 22}
    #mpl.rc('font', **font)
    plt.rc('text', usetex=True)
    figure = plt.figure(figsize = (6, 6.0), dpi = 150)
    figure.subplots_adjust(hspace = 0.1)
    fig1 = plt.subplot(111)
    
    for i,o in enumerate( PAdiff_vals ):
        x_pos = 9.5
        if Names[i] == "AAAA":
            plt.text(x_pos-0.3, 2*i-0.52, Names[i].replace("_"," "), color='gray', fontsize = 20)
            plt.errorbar(o, 2*i, xerr =PAdiff_unc[i], yerr = 0, color = 'gray', label = '', marker = 'o', markeredgecolor = 'gray', ms = 8., capsize = 0, linestyle = 'None', fmt = '', zorder = 3)
        else:
            plt.text(x_pos-0.3, 2*i-0.52, Names[i].replace("_"," "), color='black', fontsize = 20)
            plt.errorbar(o, 2*i, xerr =PAdiff_unc[i], yerr = 0, color = 'k', label = '', marker = 'o', markeredgecolor = 'k', ms = 8., capsize = 0, linestyle = 'None', fmt = '', zorder = 3)
 
    #plt.scatter([0,1,2], [1,2,3], label=['3C111','3C120','3C273'])
    fig1.axvline(x = wa_dPA_mean, linewidth = 2, color = 'r', linestyle = '-')
    fig1.axvline(x = wa_dPA_mean + wa_dPA_std, linewidth = 2, color = 'r', linestyle = '--')
    fig1.axvline(x = wa_dPA_mean - wa_dPA_std, linewidth = 2, color = 'r', linestyle = '--')
    fig1.axvline(x = 0.0, linewidth = 1, color = 'gray', linestyle = (0, (5, 10)))
    plt.xlabel(r'$EVPA_{\rm rbpl} - EVPA_{\rm cat}$ (deg)')
    #plt.ylabel('Name ')
    plt.ylim([-1,2*len(PAdiff_vals)-0.6])
    plt.xlim([-3.6,8.9])
    #fig1.get_yaxis().set_label_coords(-0.07,0.5)
    fig1.get_yaxis().set_ticks([])
    #plt.xticks([-2,-1,0,1,2,3])
    #plt.legend()
    plt.savefig('PA_diff.pdf', bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()
    return

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights = weights)
    #variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    std = np.std(values - average)
    print("standard deviation = ",std)
    return (average, std/math.sqrt(len(values)))

def calcRot(stars_data):
    PAdiff_vals = []
    PAdiff_unc  = []
    Names = []
    for st in stars_data:
        if np.isnan(st.PAdiff_val) or np.isnan(st.PAdiff_unc):
            continue

        Names.append(st.name)
        PAdiff_vals.append(st.PAdiff_val)
        PAdiff_unc.append(st.PAdiff_unc)

    wa_dPA_mean, wa_dPA_std = weighted_avg_and_std(PAdiff_vals, list(map(lambda x: 1.0/x, PAdiff_unc)))
    print( "PAinstr - PAcat = ", wa_dPA_mean, " pm ", wa_dPA_std)
    return wa_dPA_mean, wa_dPA_std

def output(wa_dPA_mean, wa_dPA_std):
    fop = open('EVPA_zp.txt','w')
    fop.write("#wa_dPA_mean wa_dPA_std\n")
    fop.write(str(wa_dPA_mean) + ' ' + str(wa_dPA_std))
    fop.close()

def getStars():
    # high polarization standards
    # from https://arxiv.org/pdf/2307.06151
    stars = ["BD+57.2615",
             "BD+59.389",
             "BD+64.106",
             "CMaR1_24",
             "CygOB2_14",
             "HD147283",
             "HD147343",
             "HD150193",
             "HD154445",
             "HD155197",
             "HD161056",
             "HD183143",
             "HD204827",
             "HD215806",
             "HD236633",
             "Hiltner960",
             "VICyg12",
             "ST_B_1725+1152_11",
             "ST_B_1725+1152_35",
             "ST_L_109_381",
             "ST_L_110_233",
             "ST_L_111_1965"]

    return stars

if __name__ == "__main__":
    stars_data = []
    stars = getStars()
    for star in stars:
        st = Star(star)
        data_R = ReadData('R', star)
        corr_data_R = CorrInstPol(data_R)
        st = get_wm(corr_data_R, st)
        stars_data.append(st)

    wa_dPA_mean, wa_dPA_std = calcRot(stars_data)
    output(wa_dPA_mean, wa_dPA_std)
    plot(wa_dPA_mean, wa_dPA_std, stars_data)




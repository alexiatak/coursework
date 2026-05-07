#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./MNRAS_Style.mplstyle')
import os
import matplotlib as mpl
import math
from uncertainties import ufloat
from uncertainties import umath
from astropy import stats
from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm # to colormap 3D surfaces from blue to red
import numpy, scipy, scipy.optimize
import pickle
import stand_pol
import random
from collections import OrderedDict

DEV_LIMIT = 3 # factor of std_deviation from weighted mean

Deattachments = [2460676.5,2461041.5]
years = ['2025']

graphWidth = 920 # units are pixels
graphHeight = 600 # units are pixels
numberOfContourLines = 16

class Obs():
    def __init__(self):
        self.name = ''
        self.jd = 0
        self.PA    = 0
        self.PAerr = 0
        self.PD    = 0
        self.PDerr = 0
        self.Q    = 0
        self.Qerr = 0
        self.U    = 0
        self.Uerr = 0
        self.x = 0
        self.y = 0

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

class Extra():
    def __init__(self):
        self.JD = 0
        self.AZ = 0
        self.EL = 0
        self.AIRTEMP = 0
        self.AIRMASS = 0
        self.SMFOCUS = 0
        self.OBJECT  = 0

def readData(st):
    obs = []

    fname = os.path.join('../../../0_raw_data/standarts', 'standardsI_raw.dat')

    fop = open(fname)
    for line in fop.readlines():
        if line.startswith("#"):
            continue
        sl = line.split()
        star = sl[0]

        if star == st:
            stand = Stand(st)
            o = Obs()
            o.name  = star
            o.jd    = float(sl[1])
            o.PD    = float(sl[2])
            o.PDerr = float(sl[3])
            o.PA    = float(sl[4])
            o.PAerr = float(sl[5])
            q_rbpl = ufloat(float(sl[6]),float(sl[7]))
            q      = q_rbpl - stand.q
            u_rbpl = ufloat(float(sl[8]),float(sl[9]))
            u      = u_rbpl - stand.u
            o.Q    = q.n
            o.Qerr = q.s
            o.U    = u.n
            o.Uerr = u.s
            x0 = float(sl[10])
            x1 = float(sl[12])
            o.x = (x0 + x1)/2.
            y2 = float(sl[15])
            y3 = float(sl[17])
            o.y = (y2 + y3)/2.
            
            if o.PDerr > 0.2:
                continue
            obs.append(o)

    fop.close()
    if 0 < len(obs):
        return obs
    else:
        return None

def getExtraParam(jd, name, extra):
    dts = []
    for e in extra:
        if e.OBJECT == name:
            dts.append([abs(e.JD - jd), e])
    
    dts.sort()
    if dts[0][0] < 0.1:
        parameter = dts[0][1].EL
    return parameter

def func(data, a, b, c, d, e, f):
    #x = data[0]
    #y = data[1]
    #return a*(x**2) + b*(y**2) + c*(x*y)
    x = data[0]
    y = data[1]
    return a*(x**2) + b*(y**2) + c*(x*y) + d*x + e*y + f


def plotQU(names, JDs, xs, ys, Qs, Qerrs, Us, Uerrs, spn):
    JD_LL = Deattachments[spn]
    JD_UL = Deattachments[spn+1]
    year = years[spn]
    
    #font = {'size'   : 22}
    #mpl.rc('font', **font)
    #mpl.rc('text', usetex=False)
    plt.rc('text', usetex=True)
    figure = plt.figure(figsize=(12.0, 6.0), dpi=150)
    figure.subplots_adjust(hspace=0.1)
    
    all_markers = [".",",","o","v","^","<",">","1","*","2","3"]
    
    markers = {"BD+33.2642" : random.choice(all_markers),
               "BD+40.2704" : random.choice(all_markers),
               "HD154892"   : random.choice(all_markers),
               'HD155197'   : random.choice(all_markers),
               'HD183143'   : random.choice(all_markers),
               }


    all_colors = ["#4be385", "#a01327", "#6408ea", "#893cd6", "#12d8d0"]
    colors = {"BD+33.2642" : random.choice(all_colors),
               "BD+40.2704" : random.choice(all_colors),
               "HD154892"   : random.choice(all_colors),
               'HD155197'   : random.choice(all_colors),
               'HD183143'   : random.choice(all_colors),
               }

    # for x coordinate
    ax1 = plt.subplot2grid((2,1), (0, 0))
    ax2 = plt.subplot2grid((2,1), (1, 0))
    
    for i in range(len(names)):
        jd = JDs[i]
        if JD_LL <= jd <= JD_UL:
            ax1.errorbar(xs[i], Qs[i], xerr=0, yerr=Qerrs[i], color=colors[names[i]], label=names[i], marker=markers[names[i]], markeredgecolor=colors[names[i]], ms=2., capsize=0, linestyle='None', fmt='', zorder=1)
            ax2.errorbar(xs[i], Us[i], xerr=0, yerr=Uerrs[i], color=colors[names[i]], label=names[i], marker=markers[names[i]], markeredgecolor=colors[names[i]], ms=2., capsize=0, linestyle='None', fmt='', zorder=1)
       
    ax1.get_xaxis().set_ticks([])
    #ax1.set_xlim([560,580])
    #ax2.set_xlim([560,580])
       
    ax1.set_ylabel('Q/I')
    ax2.set_ylabel('U/I')
    
    ax1.axhline(0, dashes=(1,1))
    ax2.axhline(0, dashes=(1,1))
    plt.xlabel('x')


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 2), fontsize=7)
    
    plt.savefig('QUcorr_x_'+year+'.png',bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    
    
    
    # for y coordinate
    ax1 = plt.subplot2grid((2,1), (0, 0))
    ax2 = plt.subplot2grid((2,1), (1, 0))
    
    for i in range(len(names)):
        ax1.errorbar(ys[i], Qs[i], xerr=0, yerr=Qerrs[i], color=colors[names[i]], label=names[i], marker=markers[names[i]], markeredgecolor=colors[names[i]], ms=2., capsize=0, linestyle='None', fmt='', zorder=1)
        ax2.errorbar(ys[i], Us[i], xerr=0, yerr=Uerrs[i], color=colors[names[i]], label=names[i], marker=markers[names[i]], markeredgecolor=colors[names[i]], ms=2., capsize=0, linestyle='None', fmt='', zorder=1)
       
    ax1.get_xaxis().set_ticks([])
    #ax1.set_xlim([560,580])
    #ax2.set_xlim([560,580])
       
    ax1.set_ylabel(r'$Q/I$')
    ax2.set_ylabel(r'$U/I$')
    
    ax1.axhline(0, dashes=(1,1))
    ax2.axhline(0, dashes=(1,1))
    plt.xlabel('y')


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 2), fontsize=7)
    
    plt.savefig('QUcorr_y_'+year+'.png',bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


def SurfacePlot(func, data, fittedParameters):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)

    mpl.pyplot.grid(True)
    axes = Axes3D(f)
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = np.linspace(min(x_data), max(x_data), 20)
    yModel = np.linspace(min(y_data), max(y_data), 20)
    X, Y = np.meshgrid(xModel, yModel)

    Z = func(np.array([X, Y]), *fittedParameters)

    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)

    axes.scatter(x_data, y_data, z_data, zorder=3) # show data along with plotted surface

    axes.set_title('Surface Plot (click-drag with mouse)') # add a title for surface plot
    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label
    axes.set_zlabel('Z Data') # Z axis data label
    
    #plt.show()
    #plt.savefig('U_vs_xy.png',bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()

def ContourPlot(func, data, fittedParameters, Stokes, year):
    plt.rc('text', usetex=True)
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]
    
    
    norm = mpl.colors.Normalize(vmin=min(z_data), vmax=max(z_data))
    cmap = cm.seismic
    
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    Z = func(numpy.array([X, Y]), *fittedParameters)
   
    for i in range(len(z_data)):
        axes.plot(x_data[i], y_data[i], 'o', color=m.to_rgba(z_data[i]))

    plt.colorbar(m, ax=axes, label=Stokes+"/I", orientation="vertical")
    #axes.set_title('Contour Plot') # add a title for contour plot
    axes.set_xlabel('x (pix)') # X axis data label
    axes.set_ylabel('y (pix)') # Y axis data label

    CS = mpl.pyplot.contour(X, Y, Z, numberOfContourLines, colors='k')
    mpl.pyplot.clabel(CS, inline=1, fmt='%1.4f', fontsize=8) # labels for contours

    #plt.show()
    plt.savefig(Stokes+'_vs_xy_'+year+'.eps', bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()

def plotCorr(all_st, spn):
    JD_LL = Deattachments[spn]
    JD_UL = Deattachments[spn+1]
    year = years[spn]
    
    names = []
    JDs   = []
    Qs    = []
    Qerrs = []
    Us    = []
    Uerrs = []
    xs = []
    ys = []
    for obs in all_st:
       for ob in obs:
           if JD_LL <= ob.jd <= JD_UL:
               names.append( ob.name ) 
               xs.append( ob.x )
               ys.append( ob.y )
               JDs.append(ob.jd)
               Qs.append(ob.Q)
               Qerrs.append(ob.Qerr)
               Us.append(ob.U)
               Uerrs.append(ob.Uerr)

    # for Q
    xData = np.array(xs)
    yData = np.array(ys)
    zData = np.array(Qs)
    print("Q before=", np.std(Qs))
    
    data = [xData, yData, zData]
    
    initialParameters = [ 4.4e-06,  4.5e-06, -9.0e-06,  1.1e-01, 1.0e-02, 0.01] # these are the same as scipy default values in this example

    # here a non-linear surface fit is made with scipy's curve_fit()
    fittedParameters, pcov = scipy.optimize.curve_fit(func, [xData, yData], zData, p0 = initialParameters)
       
    #SurfacePlot(func, data, fittedParameters)
    ContourPlot(func, data, fittedParameters, "Q", year)
    
    QmodelPredictions = func(data, *fittedParameters) 
    Qcorrected = zData - QmodelPredictions

    # serialize fit object
    with open('fit_Q'+year+'.pickle', 'wb') as f:
        pickle.dump([fittedParameters, pcov], f) # , pickle.HIGHEST_PROTOCOL

    # for U
    xData = np.array(xs)
    yData = np.array(ys)
    zData = np.array(Us)
    
    print("U before=",np.std(Us))
    print('fitted prameters', fittedParameters)
    
    data = [xData, yData, zData]
    
    initialParameters = [ 1.2e-06,  1.2e-06, -2.3e-06, -1.0e-01, 1.0e-02, 0.01] # these are the same as scipy default values in this example

    # here a non-linear surface fit is made with scipy's curve_fit()
    fittedParameters, pcov = scipy.optimize.curve_fit(func, [xData, yData], zData, p0 = initialParameters)
    print('fitted prameters', fittedParameters)
       
    #SurfacePlot(func, data, fittedParameters)
    ContourPlot(func, data, fittedParameters, "U", year)
    
    UmodelPredictions = func(data, *fittedParameters) 
    Ucorrected = zData - UmodelPredictions
    
    # serialize fit object
    with open('fit_U'+year+'.pickle', 'wb') as f:
        pickle.dump([fittedParameters, pcov], f) # , pickle.HIGHEST_PROTOCOL

    print("Q after=",np.std(Qcorrected))
    print("U after=",np.std(Ucorrected))
    
    # output standard deviation of corrected standards
    with open('std_Q'+year+'.pickle', 'wb') as f:
        pickle.dump(np.std(Qcorrected), f)

    with open('std_U'+year+'.pickle', 'wb') as f:
        pickle.dump(np.std(Ucorrected), f)
    
    X = np.stack((Qcorrected, Ucorrected), axis=0)
    covar = np.cov(X)
    print("############### covariance ####################")
    print(2 * np.sqrt(np.abs(covar[0][1])) / (np.std(Qcorrected) + np.std(Ucorrected)))
    print("############### covariance ####################")
    
    print('correlation=',np.corrcoef(Qcorrected, Ucorrected))
    
    # plot both Q and U
    print(names)
    plotQU(names, JDs, xs, ys, Qcorrected, Qerrs, Ucorrected, Uerrs, spn)

if __name__ == "__main__":
    all_st = []
    addit = []
    
    
    for st in [ 'BD+33.2642' ,'BD+40.2704', 'HD154892'  ,'HD155197' ,'HD183143' ]:   



        st_data = readData(st)
        if st_data is not None:
            all_st.append( st_data )

    for spn in range(len(years)):
        plotCorr(all_st, spn)



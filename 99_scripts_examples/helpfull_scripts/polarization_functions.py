from scipy.optimize import curve_fit
import numpy as np

from scipy import optimize, interpolate
from scipy.special import erf
from scipy.optimize import minimize
import scipy.integrate as integrate
from scipy.special import erfinv

from astropy.io import fits
from astropy.wcs import wcs

import os, sys

def get_biased_p_Vaillancourt(pd, sigma_p, fill_values = np.nan):
    '''Takes a de-biased polarization fraction, and removes
       the correction to get the original, biased value. 
       The formula used is p_d = sqrt(p^2-sigma_p^2) (see Vaillancourt 2007)
       For some threshold p/sigma_p the p_d is usually set to 0.
       
       Works with arrays too.
       
       Option fill_values is the number that is returned in place of p in the 
              case that pd is zero (default: nan)
              
              
              
      Examples:
          
      1)  Input array:
          pd = np.random.uniform(0.0001,0.0002,5)
          sp = np.random.uniform(0.00002,0.0001,5)
          # set a value to 0
          pd[2]=0
          
          >>> get_biased_p_Vaillancourt(pd,sp, fill_values = np.zeros_like(pd))
          >>> array([0.00020036, 0.00013045, 0.        , 0.00014775, 0.00016685])
    
      2)  Provide (q^2+u^2)^(1/2) as fill values
          q = 0.1
          u = -0.04
          sp = 0.1
          pd = 0.
          get_biased_p_Vaillancourt(pd,sp, fill_values = np.sqrt(q**2+u**2))
    '''
    
    if isinstance(pd, np.ndarray):
        p = np.zeros_like(pd)+fill_values
        mask_zeros = pd != 0.
        p[mask_zeros] = np.sqrt( np.power(pd[mask_zeros],2) + np.power(sigma_p[mask_zeros],2) )
        
        return p
    
    elif isinstance(pd, int) or isinstance(pd,float):
        if float(pd) == 0.:
            return fill_values
        else:
            return np.sqrt(pd**2+sigma_p**2)
    
    else:
        raise TypeError('Input should be float, int or numpy array')
        
    return

def conf_lims_MAS(pmassnr, sigma = 1):
    '''
	Compute confidence limits on p_MAS (debiased polarization fraction)
        Input: ratio of p_mas/sigma_p (debiased p over uncertainty)
        Output: (default) 68% confidence limits according to Plaszczynski+ (2014)
        	Returns limits on p_true/sigma_p
        	sigma = 1 returns 1 sigma limits
              	      = 2 returns 2 sigma
              	      = 3 returns 3 sigma
    '''
    if sigma == 1:
        pa = [1,1]
        gamma = [0.6,2.01]
        beta = [0.72,0.97]
        w = -0.83
        phi = 4.41
        
    elif sigma == 2:
        pa = [1.64,1.64]
        gamma = [0.68,2.25]
        beta = [0.88,0.31]
        w = 2.03
        phi = -0.76
        
    elif sigma == 3:
        pa = [1.95,1.95]
        gamma = [0.48,2.54]
        beta = [0.56,0.22]
        w = 1.79
        phi = -1.03
        
    psnr_min = pmassnr-pa[0]*(1+beta[0]*np.exp(-gamma[0]*pmassnr)*np.sin(w*pmassnr+phi))
    psnr_max = pmassnr+pa[1]*(1-beta[1]*np.exp(-gamma[1]*pmassnr))
        
    return psnr_min, psnr_max
    

def get_p_mas(p,sigma):
    '''Compute Plaszczynski point estimator for p (debiasing method)
	Input: polarization fraction, uncertainty
	Output: p_MAS (debiased polarization fraction)
    '''
    phat = p-sigma**2*(1-np.exp(-p**2/sigma**2))/(2.*p)
    return phat

def EVPA_pdf(theta,P0):
    """
    Compute PDF of EVPA, given the EVPA in radians and the polarization fraction 
    EVPA measurements are also non-Gaussian and defined by the following
    probability density (Naghizadeh-Khouei & Clarke 1993):
    """
    if isinstance(P0, int):
        P0 = float(P0)
    elif isinstance(P0, np.ndarray):
        P0 = P0.astype(float)
    g = 1/np.sqrt(np.pi)
    ita0 = P0/np.sqrt(2) * np.cos(2 * theta)
    g = g * (g + ita0 * np.exp(ita0**2) * (1 + erf(ita0)))
    g = g * np.exp(-(P0**2)/2)
    return g

def int_eq(sigma,snr):
    """ This is the integral of EVPA probability density from -sigma to sigma """
    integ = integrate.quad(lambda x: EVPA_pdf(x,snr),-sigma,sigma)
    return abs(integ[0] - 0.68268949)

def getSigma(pd,pd_err, estimator = 'MAS'):
    """
    Compute the uncertainty of EVPA, given polarization fraction and its uncertainty
    """
    snr = pd/pd_err
    if snr > 20:
        # it is a good approximation even for snr = 5
        return 0.5*1.0/snr

    # Use the asymptotic estimator sqrt{p^2-s^2} as in Vaillancourt
    if estimator == 'AS':
        if snr < np.sqrt(2.0):
            pd = 0.0
        else:
            pd = np.sqrt(pd**2 - pd_err**2)
            
        snr = pd/pd_err
    # Use the modified asymptotic estimator defined by Plaszczynski+ 2014. 
    # Better at low snr, equally good at high snr
    elif estimator == 'MAS':
        pmas = get_p_mas(pd,pd_err)
        snr = pmas/pd_err
    

    res = minimize(int_eq, [np.pi/50], args=(snr,), method='Nelder-Mead', tol=1e-5) # np.pi/50 = 3.6 deg - just a reasonable guess
    if res.status != 0:
        print('Something is wrong with the EVPA uncertainty calculation:\n')
        print(' '+res.message+'\n')
        return np.nan

    return res.x[0]

def p_EVPA_from_qu(q,u,sq,su):
    '''
    Compute polarization fraction and EVPA and uncertainties given q, u and their uncertainties
    '''
    p = np.sqrt(q**2 + u**2)
    s_p = np.sqrt((q**2*sq**2 + u**2*su**2)/(q**2+u**2))
    evpa = np.arctan2(u,q)/2.0 
    s_evpa = getSigma(p,s_p)

    return p, s_p, evpa, s_evpa




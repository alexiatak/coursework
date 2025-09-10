import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

def gal_to_icrs(ls,bs):
    """ Accepts lists of galactic longitudes and latitudes.
        Returns lists of RA, Dec
    """
    stars = SkyCoord(l=ls, b = bs, frame= 'galactic', unit = 'deg')
    
    return stars.icrs.ra.deg, stars.icrs.dec.deg

def icrs_to_gal(ra, dec):
    """ Accepts lists of ra and dec in degrees
        Returns lists of l and b
    """
    
    stars = SkyCoord(ra = ra,dec =dec, frame = 'icrs', unit = 'deg')
    return stars.galactic.l.deg, stars.galactic.b.deg


def convert_angle_range(angles):
    """ Convert angles from range [-pi,pi] or [-pi/2,pi/2]  
        to [0,pi).
    """
    for aa in range(len(angles)):
        if angles[aa] < 0:
            angles[aa] = np.pi - abs(angles[aa])
        if angles[aa] == np.pi:
            angles[aa] = 0
        if angles[aa] > np.pi:
            angles[aa] = angles[aa] - np.pi

    return np.array(angles)


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
    l = np.array(l) 
    b = np.array(b)
    evpa = np.array(evpa)
    Dt = np.arctan2(np.sin(np.radians(lp-l)),\
        (np.cos(np.radians(b))*np.tan(np.radians(bp))-\
            np.cos(np.radians(lp-l))*np.sin(np.radians(b))))
    return evpa + Dt

#!/usr/bin/python

import os
import math
import numpy as np
import matplotlib.pyplot as plt


DATA_FOLD = './standards'


class Measurement():
    def __init__(self):
        self.star           = None
        self.fits_file      = None
        self.full_fits_file = None
        self.dir_path       = None
        self.dir_id         = None
        self.opt_ap         = None
        self.jd             = None
        self.q              = None
        self.sq             = None
        self.u              = None
        self.uq             = None
        self.PD             = None
        self.sPD            = None
        self.EVPA           = None
        self.sEVPA          = None
        self.x0             = None
        self.y0             = None
        self.x1             = None
        self.y1             = None
        self.x2             = None
        self.y2             = None
        self.x3             = None
        self.y3             = None


def readStNames():
    star_names = []
    fop = open('uniq_st_names.txt','r')
    for line in fop.readlines():
        if line.startswith("#"):
            continue
        star_names.append(line.rstrip('\n'))
    fop.close()
    return star_names

def plot(data):
    # data contains all optimal measurements for a single star
    dir_ids = []
    for obs in data:
        dir_ids.append(obs.dir_id)

    uniq_dirs = list(set(dir_ids))
    for dir_id in uniq_dirs:
        qs  = []
        sqs = []
        us  = []
        sus = []
        xs  = []
        sxs = []
        ys  = []
        sys = []
        fits_files = []
        for obs in data:
            if dir_id == obs.dir_id:
                qs.append(obs.q)
                sqs.append(obs.sq)
                us.append(obs.u)
                sus.append(obs.su)
                xs.append((obs.x0 + obs.x1)/2.)
                sxs.append(abs(obs.x0 - obs.x1))
                ys.append((obs.y2 + obs.y3)/2.)
                sys.append(abs(obs.y2 - obs.y3))
                fits_files.append(obs.fits_file)


        # plotting q-u plane
        im_path_qu = os.path.join( DATA_FOLD, dir_id, 'q_u.png' )
        plt.figure()
        for i in range(len(qs)):
            plt.errorbar(qs[i], us[i], xerr=sqs[i], yerr=sus[i], fmt='o', label=fits_files[i])

        plt.xlabel('q')
        plt.ylabel('u')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
        plt.savefig(im_path_qu, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

        # plotting x-y plane
        im_path_xy = os.path.join( DATA_FOLD, dir_id, 'x_y.png' )
        plt.figure()
        for i in range(len(qs)):
            plt.errorbar(xs[i], ys[i], xerr=sxs[i], yerr=sys[i], fmt='o', label=fits_files[i])

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
        plt.savefig(im_path_xy, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()
    return

def readData(star_name):
    data = []
    # reading all data in subfolders like  ./standards/BD+28.4211_1  ./standards/BD+28.4211_2
    # finding thier names first
    files = os.listdir(DATA_FOLD)
    files = filter(lambda x: x.startswith(star_name), files)
    dirs = (fil for fil in files if os.path.isdir(os.path.join(DATA_FOLD, fil)))
    dirs = sorted(list(dirs), key=lambda a: a)

    for dr in dirs:
        # reading apertures.txt in each subfolder
        ap_file = os.path.join(DATA_FOLD, dr, 'apertures.txt')
        fop = open(ap_file)
        for line in fop.readlines():
            if line.startswith("#"):
                continue
            sl = line.split()
            obs                = Measurement()
            obs.star           = star_name
            obs.fits_file      = sl[0]
            obs.full_fits_file = os.path.join(DATA_FOLD, dr, obs.fits_file)
            obs.dir_path       = os.path.join(DATA_FOLD, dr)
            obs.dir_id         = dr
            obs.opt_ap         = sl[1].rstrip('\n') # reading as string
            data.append(obs)
        fop.close()

        # now iterating among folders corresponding to individual fits files and reading the
        # polarization parameters at optimal apertures
        for obs in data:
            # Another option is to read these data from the pickle
            fop = open(os.path.join(obs.full_fits_file, obs.star + '_result.dat'))
            fop.readline()
            jd_line = fop.readline()
            obs.jd = float(jd_line.split("#")[1])
            for line in fop.readlines():
                if line.startswith("#"):
                    continue
                sl = line.split()
                if sl[0] == obs.opt_ap:
                    # this is the optimal aperture parameters
                    obs.q     = float(sl[1])
                    obs.sq    = float(sl[2])
                    obs.u     = float(sl[3])
                    obs.su    = float(sl[4])
                    obs.PD    = float(sl[5])
                    obs.sPD   = float(sl[6])
                    obs.EVPA  = float(sl[7])
                    obs.sEVPA = float(sl[8])
                    obs.x0    = float(sl[9])
                    obs.y0    = float(sl[10])
                    obs.x1    = float(sl[11])
                    obs.y1    = float(sl[12])
                    obs.x2    = float(sl[13])
                    obs.y2    = float(sl[14])
                    obs.x3    = float(sl[15])
                    obs.y3    = float(sl[16])
            fop.close()
            if obs.PD is None:
                print("Problems with ", obs.star, obs.fits_file, obs.opt_ap)

    # plotting q-u plane to find and exclude outliers
    plot(data)
    return data

def outputStar(data_individ_star):
    header = '#Name   JD    P   sP    PA[deg]  sPA[deg]   q        sq       u        su   x0    y0    x1   y1   x2    y2    x3    y3\n'

    OUT_PATH = os.path.join( os.getcwd(), 'standards', data_individ_star[0].star +'_raw.dat' )
    fop_out = open(OUT_PATH,'w')
    fop_out.write(header)
    for obs in data_individ_star:
        out_str = '{star: <16}'.format(star=obs.star)
        out_str += "{:13.5f}".format(obs.jd)
        out_str += "{:10.5f}".format(obs.PD)
        out_str += "{:10.5f}".format(obs.sPD)
        out_str += "{:8.2f}".format(obs.EVPA)
        out_str += "{:8.2f}".format(obs.sEVPA)
        out_str += "{:9.5f}".format(obs.q)
        out_str += "{:9.5f}".format(obs.sq)
        out_str += "{:9.5f}".format(obs.u)
        out_str += "{:9.5f}".format(obs.su)
        out_str += "{:9.2f}".format(obs.x0)
        out_str += "{:9.2f}".format(obs.y0)
        out_str += "{:9.2f}".format(obs.x1)
        out_str += "{:9.2f}".format(obs.y1)
        out_str += "{:9.2f}".format(obs.x2)
        out_str += "{:9.2f}".format(obs.y2)
        out_str += "{:9.2f}".format(obs.x3)
        out_str += "{:9.2f}".format(obs.y3)
        out_str += "\n"
        fop_out.write(out_str)
    fop_out.close()

def outputAll(data):
    header = '#Name   JD    P   sP    PA[deg]  sPA[deg]   q        sq       u        su   x0    y0    x1   y1   x2    y2    x3    y3\n'

    OUT_PATH = os.path.join( os.getcwd(), 'standards_raw.dat' )
    fop_out = open(OUT_PATH,'w')
    fop_out.write(header)
    for obs in data:
        out_str = '{star: <16}'.format(star=obs.star)
        out_str += "{:13.5f}".format(obs.jd)
        out_str += "{:10.5f}".format(obs.PD)
        out_str += "{:10.5f}".format(obs.sPD)
        out_str += "{:8.2f}".format(obs.EVPA)
        out_str += "{:8.2f}".format(obs.sEVPA)
        out_str += "{:9.5f}".format(obs.q)
        out_str += "{:9.5f}".format(obs.sq)
        out_str += "{:9.5f}".format(obs.u)
        out_str += "{:9.5f}".format(obs.su)
        out_str += "{:9.2f}".format(obs.x0)
        out_str += "{:9.2f}".format(obs.y0)
        out_str += "{:9.2f}".format(obs.x1)
        out_str += "{:9.2f}".format(obs.y1)
        out_str += "{:9.2f}".format(obs.x2)
        out_str += "{:9.2f}".format(obs.y2)
        out_str += "{:9.2f}".format(obs.x3)
        out_str += "{:9.2f}".format(obs.y3)
        out_str += "\n"
        fop_out.write(out_str)
    fop_out.close()

if __name__ == "__main__":
    data = []
    star_names = readStNames()
    for star_name in star_names:
        print(star_name)
        data_individ_star = readData(star_name)
        outputStar(data_individ_star)
        data += data_individ_star
    outputAll(data)














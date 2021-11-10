import sys
import os
import glob
import time
import copy
import warnings
import requests
import subprocess

from itertools import cycle
from shutil import copyfile

import numpy as np
import pandas as pd
import numpy.lib.recfunctions as rfn

import scipy.spatial
import scipy.optimize
import scipy.interpolate

import astropy.io.fits as fitsio
import astropy.io.ascii as ascii
import astropy.units as u
import astropy.wcs.utils
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Ellipse, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import esutil
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

from astropy.wcs import WCS
from astropy.cosmology import Planck15
from PIL import Image
from joblib import Parallel, delayed
from collections import OrderedDict

light = 3e18  # Angs / s
sdss_filters = ["u", "g", "r", "i", "z"]
detimg_filters = ["u", "g", "r", "i"]

sdss_filters_pivot = {"u": 3594.9, "g": 4640.4, "r": 6122.3, "i": 7439.5, "z": 8897.1}
hst_filters_pivot = {"F435W":4329.17,"F606W":5921.94,"F775W":7693.18,"F850LP":9033.22,"F125W":12486.07}

sdss_filters_width = {"u": 540.97, "g": 1262.68, "r": 1055.51, "i": 1102.57, "z": 1164.01}
hst_filters_width = {"F435W":939.00,"F606W":1771.39,"F775W":1378.60,"F850LP":1324.61,"F125W":2674.41}

JPEG_pixscale = 0.12
FITS_pixscale = 0.396
S82_ZP = 30.

cwd = "/data/highzgal/mehta/Clumps/S82_analysis/"

def matchRADec(ra1, dec1, ra2, dec2, crit, maxmatch=0):
    """
    Matches two catalogs by (ra,dec)
    """
    h = esutil.htm.HTM(10)
    crit = crit / 3600.0  # crit arcsec --> deg
    m1, m2, d12 = h.match(ra1, dec1, ra2, dec2, crit, maxmatch=maxmatch)
    return m1, m2, d12

def getAbsMagFromAppMag(app_mag,z):
    """
    Returns the absolute magnitude for a given apparent magnitude at redshift z.
    """
    dist = Planck15.luminosity_distance(z).to(u.pc).value
    if isinstance(app_mag,np.ndarray):
        cond = (np.abs(app_mag)!=99.)
        abs_mag = np.zeros(len(app_mag)) + 99.
        abs_mag[cond] = app_mag[cond] - 5*(np.log10(dist[cond]) - 1) + 2.5*np.log10(1+z[cond])
    else:
        abs_mag = app_mag - 5*(np.log10(dist) - 1) + 2.5*np.log10(1+z) if np.abs(app_mag)!=99. else app_mag
    return abs_mag

def calcFluxScale(zp0,zp1):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

def calcCoM(img):
    """
    Calculate the center of mass for a given image
    """
    y, x = np.indices(img.shape) + 1
    xc = np.ma.sum(img * x) / np.ma.sum(img)
    yc = np.ma.sum(img * y) / np.ma.sum(img)
    return xc, yc

def gauss2d(x,y,sig,x0=0,y0=0):

    return 0.5/sig**2/np.pi * np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sig**2)

def fwhmToSigma(fwhm):

    return fwhm / (2 * np.sqrt(2*np.log(2.)))

def gauss1d(x,flux,x0,sig):

    return flux * 1/np.sqrt(2*np.pi*sig**2) * np.exp(-0.5*(x-x0)**2/sig**2)

def continuum(x,x0,a):

    return a #+ b*(x-x0)

def emLineModel(x,flux,x0,sig,cont):

    return gauss1d(x,flux,x0,sig) + continuum(x,x0,cont)

def getApertureCorr(psf,aper):

    dx = dy = 0.01
    x = np.arange(-10,10,dx)
    y = np.arange(-10,10,dy)

    yy,xx = np.meshgrid(y,x,sparse=True)
    g_see = gauss2d(xx,yy,sig=fwhmToSigma(psf))

    cond_aper = np.sqrt(xx**2 + yy**2) <= aper
    f_see = np.sum(g_see[cond_aper]) * dx * dy

    aper_adjust = 1/f_see
    return aper_adjust

def runBashCommand(call, cwd, verbose=True):
    """
    Generic function to execute a bash command
    """
    start = time.time()
    if isinstance(verbose, str):
        f = open(verbose, "w")
        p = subprocess.Popen(call, stdout=f, stderr=f, cwd=cwd, shell=True)
    elif verbose == True:
        print("Running command:<{:s}> in directory:<{:s}> ... ".format(call, cwd))
        p = subprocess.Popen(
            call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, shell=True
        )
        for line in iter(p.stdout.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    else:
        devnull = open(os.devnull, "w")
        p = subprocess.Popen(call, stdout=devnull, stderr=devnull, cwd=cwd, shell=True)
    p.communicate()
    p.wait()
    return time.time() - start

def viewFields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to keep.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = a.view(newdt)
    return b

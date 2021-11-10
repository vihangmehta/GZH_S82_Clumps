import numpy as np
import astropy.io.fits as fitsio

from useful import os, cwd
from utils import getTotalNClumps

idOffset = 1e4

def mkClumpInput(sample,savename,fluxtype,destdir="boxcar10"):

    dtype = [("ID",int),("gal_id",int),("clump_id",int),("zspec",float)]
    for filt in "ugriz": dtype.extend([("flux_sdss_%s"%filt,float),("fluxerr_sdss_%s"%filt,float)])

    N = getTotalNClumps(sample=sample, destdir=destdir)
    beagle_input = np.recarray(N,dtype=dtype)

    i = 0
    for j, entry in enumerate(sample):

        try:
            clumps = fitsio.getdata("../photom/{1:s}/objID{0[survey_id]:020d}-phot.fits".format(entry, destdir))
        except OSError:
            continue

        beagle_input[i:i+len(clumps)]["ID"]       = (j+1)*idOffset + clumps["ID"]
        beagle_input[i:i+len(clumps)]["gal_id"]   = entry["survey_id"]
        beagle_input[i:i+len(clumps)]["clump_id"] = clumps["ID"]
        beagle_input[i:i+len(clumps)]["zspec"]    = entry["REDSHIFT_DR14"]

        for filt in "ugriz":

            if   fluxtype=="clump":
                fluxcol,ferrcol = "FLUX_%s"%filt,"FLUXERR_%s"%filt
            elif fluxtype=="errfix":
                fluxcol,ferrcol = "FLUX_%s"%filt,"FLUXERR_%s"%filt
                clumps["FLUXERR_%s"%filt] = np.sqrt(clumps["ORGFLUXERR_%s"%filt]**2 + clumps["UNDFLUXERR_%s"%filt]**2)
            elif fluxtype=="galsub":
                fluxcol,ferrcol = "UNDFLUX_%s"%filt,"UNDFLUXERR_%s"%filt
            else:
                raise Exception("Invalid fluxtype.")

            beagle_input[i:i+len(clumps)]["flux_sdss_%s"%filt]    = clumps[fluxcol]
            beagle_input[i:i+len(clumps)]["fluxerr_sdss_%s"%filt] = clumps[ferrcol]

        i += len(clumps)

    fitsio.writeto(savename,beagle_input,overwrite=True)

    beagle_input_split = np.array_split(beagle_input,3)
    fitsio.writeto(savename.replace(".fits",".1.fits"),beagle_input_split[0],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".2.fits"),beagle_input_split[1],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".3.fits"),beagle_input_split[2],overwrite=True)

def mkGalaxyInput(sample,savename,aper="auto"):

    photom = fitsio.getdata("../photom/galaxy/galaxy_photom.fits")

    dtype = [("ID",int),("gal_id",int),("zspec",float)]
    for filt in "ugriz": dtype.extend([("flux_sdss_%s"%filt,float),("fluxerr_sdss_%s"%filt,float)])
    beagle_input = np.recarray(len(sample),dtype=dtype)

    assert all(sample["survey_id"]==photom["survey_id"])

    beagle_input["ID"]     = np.arange(len(sample))+1
    beagle_input["gal_id"] = sample["survey_id"]
    beagle_input["zspec"]  = sample["REDSHIFT_DR14"]

    for filt in "ugriz":
        beagle_input["flux_sdss_%s"%filt]    = photom["flux_%s_%s"%(aper,filt)]
        beagle_input["fluxerr_sdss_%s"%filt] = photom["fluxerr_%s_%s"%(aper,filt)]

    fitsio.writeto(savename,beagle_input,overwrite=True)

    beagle_input_split = np.array_split(beagle_input,3)
    fitsio.writeto(savename.replace(".fits",".1.fits"),beagle_input_split[0],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".2.fits"),beagle_input_split[1],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".3.fits"),beagle_input_split[2],overwrite=True)

def mkDiffGalInput(sample,savename,destdir="diffgal"):

    dtype = [("ID",int),("gal_id",int),("clump_id",int),("zspec",float)]
    for filt in "ugriz": dtype.extend([("flux_sdss_%s"%filt,float),("fluxerr_sdss_%s"%filt,float)])

    N = getTotalNClumps(sample=sample, destdir=destdir, suffix="diffgal-phot")
    beagle_input = np.recarray(N,dtype=dtype)

    i = 0
    for j, entry in enumerate(sample):

        try:
            phot = fitsio.getdata("../photom/{1:s}/objID{0[survey_id]:020d}-diffgal-phot.fits".format(entry, destdir))
        except OSError:
            continue

        beagle_input[i:i+len(phot)]["ID"]       = (j+1)*idOffset + phot["ID"]
        beagle_input[i:i+len(phot)]["gal_id"]   = entry["survey_id"]
        beagle_input[i:i+len(phot)]["clump_id"] = phot["ID"]
        beagle_input[i:i+len(phot)]["zspec"]    = entry["REDSHIFT_DR14"]

        for filt in "ugriz":

            fluxcol,ferrcol = "FLUX_%s"%filt,"FLUXERR_%s"%filt
            beagle_input[i:i+len(phot)]["flux_sdss_%s"%filt]    = phot[fluxcol]
            beagle_input[i:i+len(phot)]["fluxerr_sdss_%s"%filt] = phot[ferrcol]

        i += len(phot)

    fitsio.writeto(savename,beagle_input,overwrite=True)

    beagle_input_split = np.array_split(beagle_input,6)
    fitsio.writeto(savename.replace(".fits",".1.fits"),beagle_input_split[0],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".2.fits"),beagle_input_split[1],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".3.fits"),beagle_input_split[2],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".4.fits"),beagle_input_split[3],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".5.fits"),beagle_input_split[4],overwrite=True)
    fitsio.writeto(savename.replace(".fits",".6.fits"),beagle_input_split[5],overwrite=True)

if __name__ == '__main__':

    sample = fitsio.getdata("../samples/final_clumpy_sample.fits")

    # mkClumpInput(sample=sample,savename="data/clumps_beagle_input.fits",fluxtype="clump")
    # mkClumpInput(sample=sample,savename="data/clumps_beagle_input.errfix.fits",fluxtype="errfix")
    # mkClumpInput(sample=sample,savename="data/clumps_beagle_input.galsub.fits",fluxtype="galsub")

    # mkGalaxyInput(sample=sample,savename="data/clumps_beagle_input.galaxy.fits",aper="auto")
    # mkGalaxyInput(sample=sample,savename="data/clumps_beagle_input.galiso.fits",aper="iso")

    mkDiffGalInput(sample=sample,savename="data/clumps_beagle_input.diffgal.fits")

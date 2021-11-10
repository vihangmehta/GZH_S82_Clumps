from useful import *
from utils_clumps import *
from utils_errors import *
from gal_extinction import Gal_Extinction
from plotter import plotClumpsPhotometry

from photutils import CircularAperture, CircularAnnulus, aperture_photometry

def setupCatalog(trimcat):

    dtype = trimcat.dtype.descr

    _dtype = [("NPIX_APER", float), ("EXTINCT_EBV", float),("EXTINCT_AV", float)]

    for filt in sdss_filters:
        _dtype.extend([("FLUX_{:s}".format(filt), float),
                       ("FLUXERR_{:s}".format(filt), float),
                       ("MAG_{:s}".format(filt), float),
                       ("MAGERR_{:s}".format(filt), float),
                       ("ORGFLUX_{:s}".format(filt), float),
                       ("ORGFLUXERR_{:s}".format(filt), float),
                       ("ORGMAG_{:s}".format(filt), float),
                       ("ORGMAGERR_{:s}".format(filt), float),
                       ("UNDFLUX_{:s}".format(filt), float),
                       ("UNDFLUXERR_{:s}".format(filt), float),
                       ("UNDMAG_{:s}".format(filt), float),
                       ("UNDMAGERR_{:s}".format(filt), float),
                       ("DIFFLUX_{:s}".format(filt), float),
                       ("DIFFSTD_{:s}".format(filt), float),
                       ("IMGBKGD_{:s}".format(filt), float),
                       ("EXTINCT_{:s}".format(filt), float)])

    catalog = np.recarray(len(trimcat),dtype=dtype+_dtype)
    for x in  dtype: catalog[x[0]] = trimcat[x[0]]
    for x in _dtype: catalog[x[0]] = -99

    return catalog

def applyAperCorr(catalog,entry):

    apersize = getClumpApertureSize(entry=entry)
    aperCorr = getApertureCorr(psf=apersize["psf_avg"],aper=0.5*apersize["phot"])

    for filt in sdss_filters:
        for x in ["ORGFLUX","UNDFLUX","FLUX"]:
            catalog["{:s}_{:s}".format(x,filt)] *= aperCorr
            catalog["{:s}ERR_{:s}".format(x,filt)] *= aperCorr
    return catalog

def removeGalacticExtinction(catalog):

    gal_ext = Gal_Extinction()
    catalog['EXTINCT_EBV'] = gal_ext.calc_EBV(catalog['RA'],catalog['DEC'])
    catalog['EXTINCT_AV'] = gal_ext.calc_Av(ebv=catalog['EXTINCT_EBV'])

    for filt in sdss_filters:
        catalog["EXTINCT_{:s}".format(filt)] = gal_ext.calc_Alambda(filt=filt,Av=catalog["EXTINCT_AV"])[0]
        for x in ["ORGFLUX","UNDFLUX","FLUX"]:
            catalog["{:s}_{:s}".format(x,filt)]    = gal_ext.remove_gal_ext(flux=catalog["{:s}_{:s}".format(x,filt)],
                                                                            filt=filt, Av=catalog["EXTINCT_AV"])
            catalog["{:s}ERR_{:s}".format(x,filt)] = gal_ext.remove_gal_ext(flux=catalog["{:s}ERR_{:s}".format(x,filt)],
                                                                            filt=filt, Av=catalog["EXTINCT_AV"])
    return catalog

def calcMagnitudes(catalog):

    for filt in sdss_filters:
        for x in ["ORGFLUX","UNDFLUX","FLUX"]:

            y = x.replace("FLUX","MAG")
            flux = catalog["{:s}_{:s}".format(x,filt)]
            fluxerr = catalog["{:s}ERR_{:s}".format(x,filt)]

            cond = (flux > 0)
            catalog["{:s}_{:s}".format(y,filt)][cond]    = (-2.5 * np.log10(flux[cond]) + S82_ZP)
            catalog["{:s}ERR_{:s}".format(y,filt)][cond] = ( 2.5 / np.log(10) * (fluxerr[cond] / flux[cond]))

    return catalog

def convertFluxTouJy(catalog):

    fluxScale = calcFluxScale(zp0=23.9,zp1=S82_ZP)

    for filt in sdss_filters:

        for x in ["ORGFLUX","UNDFLUX","FLUX"]:

            catalog["{:s}_{:s}".format(x,filt)] /= fluxScale
            catalog["{:s}ERR_{:s}".format(x,filt)] /= fluxScale

        catalog["DIFFLUX_{:s}".format(filt)] /= fluxScale
        catalog["DIFFSTD_{:s}".format(filt)] /= fluxScale
        catalog["IMGBKGD_{:s}".format(filt)] /= fluxScale

    return catalog

def measureClumpPhotometry(entry, destdir, savefig=True):

    savename = "photom/{1:s}/objID{0[survey_id]:020d}-phot.fits".format(entry, destdir)
    catname  = "photom/{1:s}/objID{0[survey_id]:020d}-trim.fits".format(entry, destdir)
    ssegname = "photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry)

    try:
        catalog = fitsio.getdata(catname)
    except OSError:
        catalog = None

    if catalog is not None:

        catalog = setupCatalog(catalog)
        apersize = getClumpApertureSize(entry=entry)
        apertures = getClumpApertures(catalog,entry=entry)
        clump_mask = getClumpMask(catalog,imshape=fitsio.getdata(ssegname).shape,radius=0.5*apersize["mask"])
        catalog["NPIX_APER"] = apertures["aper"].area

        for j,filt in enumerate(sdss_filters):

            imgname = "S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry, filt)
            img, img_hdr = fitsio.getdata(imgname, header=True)

            ### Clump photometry
            photom = aperture_photometry(img, apertures["aper"], method="subpixel", subpixels=5)
            catalog["ORGFLUX_{:s}".format(filt)] = photom["aperture_sum"]
            catalog["ORGFLUXERR_{:s}".format(filt)] = calcTsfError(entry=entry,filt=filt,flux=catalog["ORGFLUX_{:s}".format(filt)],npix=catalog["NPIX_APER"])

            ### Diffuse galaxy light calc
            mask_img = img.copy()
            mask_img[clump_mask] = -99.0
            for i, mask in enumerate(apertures["annl"].to_mask(method="center")):
                annulus = mask.multiply(mask_img)
                annulus = annulus[(annulus != -99.0) & (mask.data != 0)]
                if len(annulus) < 10:
                    raise Warning("Diffuse galaxy light determined using less than 10px for {:d} - {:s} - clump {:d}.".format(entry["survey_id"], filt, i+1))
                catalog["DIFFLUX_{:s}".format(filt)][i] = np.median(annulus)
                catalog["DIFFSTD_{:s}".format(filt)][i] = np.std(annulus)

            ### Calc the underlying galaxy light within aperture
            catalog["UNDFLUX_{:s}".format(filt)]    = catalog["DIFFLUX_{:s}".format(filt)] * catalog["NPIX_APER"]
            catalog["UNDFLUXERR_{:s}".format(filt)] = catalog["DIFFSTD_{:s}".format(filt)] * np.sqrt(catalog["NPIX_APER"])

            ### Clip when the underlying galaxy light is zero/negative or brighter than the aperture flux itself
            cond1 = (catalog["UNDFLUX_{:s}".format(filt)] <= 0)
            catalog["UNDFLUX_{:s}".format(filt)][cond1] = 0
            cond2 = (catalog["UNDFLUX_{:s}".format(filt)] >= catalog["ORGFLUX_{:s}".format(filt)])
            catalog["UNDFLUX_{:s}".format(filt)][cond2] = catalog["ORGFLUX_{:s}".format(filt)][cond2]

            ### Calc the clump flux by subtracting the galaxy flux
            catalog["FLUX_{:s}".format(filt)] = catalog["ORGFLUX_{:s}".format(filt)] - catalog["UNDFLUX_{:s}".format(filt)]
            catalog["FLUXERR_{:s}".format(filt)] = catalog["ORGFLUXERR_{:s}".format(filt)]
            # catalog["FLUXERR_{:s}".format(filt)] = np.sqrt(catalog["ORGFLUXERR_{:s}".format(filt)]**2 + catalog["UNDFLUXERR_{:s}".format(filt)]**2)

            ### Save image background level
            catalog["IMGBKGD_{:s}".format(filt)] = calcImgBackground(img=img,sseg=fitsio.getdata(ssegname))

        catalog = applyAperCorr(catalog,entry=entry)
        catalog = removeGalacticExtinction(catalog)
        catalog = calcMagnitudes(catalog)
        catalog = convertFluxTouJy(catalog)

        fitsio.writeto(savename, catalog,overwrite=True)

    else:

        print("No clumps found for objID{0[survey_id]:020d}".format(entry))
        if os.path.isfile(savename): os.remove(savename)
        os.system("touch {:s}".format(savename))
        catalog = None

    plotClumpsPhotometry(entry=entry,destdir=destdir,photcat=catalog,savefig=savefig)

def main(sample):

    Parallel(n_jobs=15,verbose=10,backend="multiprocessing")(delayed(measureClumpPhotometry)(entry,destdir="boxcar10") for entry in sample)

    filelist = " ".join(["plots/summary_clumps/objID{:020d}-photom.png".format(i) for i in sample["survey_id"]])
    savename="plots/pdf/final_clumpy_photometry.pdf"
    os.system("convert {0:s} {1:s}".format(filelist, savename))

def test(sample):

    entry = sample[sample["survey_id"] == 8647474690337538126][0]
    measureClumpPhotometry(entry, destdir="boxcar10", savefig=True)

if __name__ == "__main__":

    sample = fitsio.getdata("samples/final_clumpy_sample.fits")

    # main(sample=sample)
    test(sample=sample)

    plt.show()

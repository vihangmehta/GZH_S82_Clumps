from useful import *
from utils_clumps import *
from utils_errors import *
from gal_extinction import Gal_Extinction
from plotter import getVminVmax

from photutils import CircularAperture, CircularAnnulus, aperture_photometry

def setupCatalog(trimcat,N):

    dtype = trimcat.dtype.descr[:-1]

    _dtype = [("NPIX_APER", float), ("EXTINCT_EBV", float),("EXTINCT_AV", float)]

    for filt in sdss_filters:
        _dtype.extend([("FLUX_{:s}".format(filt), float),
                       ("FLUXERR_{:s}".format(filt), float),
                       ("MAG_{:s}".format(filt), float),
                       ("MAGERR_{:s}".format(filt), float),
                       ("EXTINCT_{:s}".format(filt), float)])

    catalog = np.recarray(N,dtype=dtype+_dtype)
    for x in  dtype:
        catalog[x[0]] = trimcat[x[0]][0] if "DIST" not in x[0] else -99
    for x in _dtype: catalog[x[0]] = -99

    catalog["ID"] = np.arange(len(catalog))+1

    return catalog

def applyAperCorr(catalog,entry):

    apersize = getClumpApertureSize(entry=entry)
    aperCorr = getApertureCorr(psf=apersize["psf_avg"],aper=0.5*apersize["phot"])

    for filt in sdss_filters:
        catalog["FLUX_{:s}".format(filt)] *= aperCorr
        catalog["FLUXERR_{:s}".format(filt)] *= aperCorr
    return catalog

def removeGalacticExtinction(catalog):

    gal_ext = Gal_Extinction()
    catalog['EXTINCT_EBV'] = gal_ext.calc_EBV(catalog['RA'],catalog['DEC'])
    catalog['EXTINCT_AV'] = gal_ext.calc_Av(ebv=catalog['EXTINCT_EBV'])

    for filt in sdss_filters:
        catalog["EXTINCT_{:s}".format(filt)] = gal_ext.calc_Alambda(filt=filt,Av=catalog["EXTINCT_AV"])[0]
        catalog["FLUX_{:s}".format(filt)]    = gal_ext.remove_gal_ext(flux=catalog["FLUX_{:s}".format(filt)],
                                                                        filt=filt, Av=catalog["EXTINCT_AV"])
        catalog["FLUXERR_{:s}".format(filt)] = gal_ext.remove_gal_ext(flux=catalog["FLUXERR_{:s}".format(filt)],
                                                                        filt=filt, Av=catalog["EXTINCT_AV"])
    return catalog

def calcMagnitudes(catalog):

    for filt in sdss_filters:

        flux = catalog["FLUX_{:s}".format(filt)]
        fluxerr = catalog["FLUXERR_{:s}".format(filt)]

        cond = (flux > 0)
        catalog["MAG_{:s}".format(filt)][cond]    = (-2.5 * np.log10(flux[cond]) + S82_ZP)
        catalog["MAGERR_{:s}".format(filt)][cond] = ( 2.5 / np.log(10) * (fluxerr[cond] / flux[cond]))

    return catalog

def convertFluxTouJy(catalog):

    fluxScale = calcFluxScale(zp0=23.9,zp1=S82_ZP)
    for filt in sdss_filters:
        catalog["FLUX_{:s}".format(filt)] /= fluxScale
        catalog["FLUXERR_{:s}".format(filt)] /= fluxScale
    return catalog

def getPhotMask(xy,imshape,radius):

    mask = np.zeros(imshape, dtype=bool)
    c, r = np.indices(mask.shape)
    cond = ((r - xy[0])**2 + (c - xy[1])**2 <= radius**2)
    mask[cond] = 1
    return mask

def getDiffGalPhotometry(entry,fullcat,clumpcat,ssegname,gsegname,plot=True,debug=False):

    ssegimg,sseghdr = fitsio.getdata(ssegname, header=True)
    gsegimg,gseghdr = fitsio.getdata(gsegname, header=True)

    apersize = getClumpApertureSize(entry=entry)
    aperarea = CircularAperture(getClumpPositions(clumpcat),r=0.5*apersize["phot"]).area
    mask = getClumpMask(clumpcat,imshape=gsegimg.shape,radius=0.5*apersize["mask"])
    # mask = getClumpMask(fullcat,imshape=gsegimg.shape,radius=0.5*apersize["mask"])

    ix,iy = int(np.round(clumpcat["GAL_XC"][0])), int(np.round(clumpcat["GAL_YC"][0]))
    ssegidx = ssegimg[iy-1,ix-1]
    gsegidx = gsegimg[iy-1,ix-1]
    cond = (gsegimg==gsegidx) & (ssegimg==ssegidx) & (~mask)

    _gsegimg = gsegimg.copy()
    _gsegimg[~cond] = -99

    c,r = np.indices(gsegimg.shape)
    dist = np.sqrt((r-clumpcat["GAL_XC"][0])**2 + (c-clumpcat["GAL_YC"][0])**2)
    theta = np.arctan2((c-clumpcat["GAL_YC"][0]), (r-clumpcat["GAL_XC"][0]))
    distnorm = dist * FITS_pixscale / clumpcat["GAL_REFF"][0]
    # cond = cond & (clumpcat["PSF_WIDTH_AVG"][0] < dist*FITS_pixscale) & (distnorm < 10)

    dr = clumpcat["PSF_WIDTH_AVG"][0] / clumpcat["GAL_REFF"][0] * 2
    rbins = np.arange(dr*0.01,max(distnorm[cond]),dr)

    catalog = None
    for r0,r1 in zip(rbins[:-1],rbins[1:]):

        condr = (r0<distnorm) & (distnorm<r1)
        da = 2.5*dr / r0
        na = max(2,int(np.ceil(2*np.pi/da)))
        abins = np.linspace(-np.pi,np.pi,na)
        cat = setupCatalog(clumpcat,N=na-1)

        for j,(a0,a1) in enumerate(zip(abins[:-1],abins[1:])):

            conda = (a0<theta) & (theta<a1)
            nValid = cond & conda & condr
            r = distnorm[conda & condr]
            a = theta[conda & condr]

            if np.sum(nValid)>aperarea:
                cat["X"][j] = np.mean(clumpcat["GAL_XC"][0] + np.cos(a) * r * clumpcat["GAL_REFF"][0] / FITS_pixscale)
                cat["Y"][j] = np.mean(clumpcat["GAL_YC"][0] + np.sin(a) * r * clumpcat["GAL_REFF"][0] / FITS_pixscale)
                cat["NPIX_APER"][j] = np.sum(nValid)
                for filt in sdss_filters:
                    imgname = "S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry, filt)
                    img, img_hdr = fitsio.getdata(imgname, header=True)
                    cat["FLUX_{:s}".format(filt)][j] = np.sum(img[nValid])

        cat = cat[cat["NPIX_APER"]>aperarea]
        catalog = rfn.stack_arrays([catalog,cat],usemask=False,asrecarray=True,autoconvert=False) if catalog is not None else cat

    ### Renumber the catalog IDs
    catalog["ID"] = np.arange(len(catalog))+1

    ### Convert the center (x,y) to (ra,dec)
    wcs = WCS(gseghdr, fix=False)
    catalog["RA"], catalog["DEC"] = wcs.all_pix2world(catalog["X"], catalog["Y"], 1)

    ### Calculate the distance from center
    catalog["DISTANCE_XY"] = np.sqrt((catalog["X"]-catalog["GAL_XC"])**2 +
                                     (catalog["Y"]-catalog["GAL_YC"])**2)
    catalog["DISTANCE"] = catalog["DISTANCE_XY"] * FITS_pixscale
    catalog["DISTNORM"] = catalog["DISTANCE"] / catalog["GAL_REFF"]
    catalog["DIST_SMA"] = catalog["DISTANCE"] / catalog["GAL_SMA"]

    if np.isfinite(entry["REDSHIFT_DR14"]):
        catalog["DISTPHYS"] = catalog["DISTANCE"] / Planck15.arcsec_per_kpc_proper(entry["REDSHIFT_DR14"]).value
    else:
        catalog["DISTPHYS"] = np.NaN

    if plot:

        fig,axes = plt.subplots(2,3,figsize=(15,10.5),dpi=75)
        fig.subplots_adjust(left=0.002,right=0.998,bottom=0.002,top=0.95,wspace=0,hspace=0)
        fig.suptitle("objID#{0[survey_id]}".format(entry),fontsize=18,fontweight=600)
        axes = axes.flatten()

        for i,filt in enumerate("ugriz"):
            img = fitsio.getdata("S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt))
            vmin,vmax = getVminVmax(img)
            axes[i].imshow(img,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)
        axes[-1].imshow(_gsegimg,cmap=plt.cm.Greys)

        for r in rbins:
            axes[-1].add_patch(Circle(xy=(img.shape[0]/2-1,img.shape[1]/2-1),radius=r*clumpcat["GAL_REFF"][0]/FITS_pixscale,lw=1,edgecolor="tab:green",facecolor="none"))

        for r0,r1 in zip(rbins[:-1],rbins[1:]):
            da = 2.5*dr / r0
            na = int(np.ceil(2*np.pi/da))
            abins = np.linspace(-np.pi,np.pi,na)
            for a in abins:
                r = np.linspace(r0,r1,3)
                x = (img.shape[0]/2) + np.cos(a) * r * clumpcat["GAL_REFF"][0] / FITS_pixscale
                y = (img.shape[0]/2) + np.sin(a) * r * clumpcat["GAL_REFF"][0] / FITS_pixscale
                axes[-1].plot(x-1,y-1,color="tab:green",lw=1)

        for ax in axes:
            ax.scatter(fullcat["X"]-1,fullcat["Y"]-1,c='tab:red',s=50,marker='x')
            ax.scatter(clumpcat["X"]-1,clumpcat["Y"]-1,c='tab:blue',s=50,marker='x',lw=2)
            ax.scatter(clumpcat["GAL_XC"][0]-1,clumpcat["GAL_YC"][0]-1,c='w',s=75,marker='+',lw=2)
            if catalog is not None:
                ax.scatter(catalog["X"]-1,catalog["Y"]-1,c='tab:green',s=75,marker='x',lw=2.5)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        fig.savefig("photom/diffgal/objID{0[survey_id]:020d}-diffgal-aper.png".format(entry))

        if debug: plt.show()

    return catalog

def measureDiffGalPhotometry(entry, destdir, debug=False):

    savename = "photom/{1:s}/objID{0[survey_id]:020d}-diffgal-phot.fits".format(entry,destdir)
    catname  = "photom/boxcar10/objID{0[survey_id]:020d}-cat.fits".format(entry)
    trimname = "photom/boxcar10/objID{0[survey_id]:020d}-trim.fits".format(entry)
    ssegname = "photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry)
    gsegname = "photom/galaxy/objID{0[survey_id]:020d}-seg.fits".format(entry)

    try:
        clumpcat = fitsio.getdata(trimname)
        fullcat  = fitsio.getdata(catname)
        fullcat  = np.array(rfn.rename_fields(fullcat, {"X_IMAGE":"X","Y_IMAGE":"Y"}))
    except OSError:
        clumpcat, fullcat = None, None

    if clumpcat is not None:

        catalog = getDiffGalPhotometry(entry=entry,fullcat=fullcat,clumpcat=clumpcat,ssegname=ssegname,gsegname=gsegname,debug=debug)

        for filt in sdss_filters:
            catalog["FLUXERR_{:s}".format(filt)] = calcTsfError(entry=entry,filt=filt,flux=catalog["FLUX_{:s}".format(filt)],npix=catalog["NPIX_APER"])

        catalog = removeGalacticExtinction(catalog)
        catalog = calcMagnitudes(catalog)
        catalog = convertFluxTouJy(catalog)

        fitsio.writeto(savename, catalog,overwrite=True)

    else:

        print("Could not place any apertures for objID{0[survey_id]:020d}".format(entry))
        if os.path.isfile(savename): os.remove(savename)
        os.system("touch {:s}".format(savename))
        catalog = None

def main(sample):

    Parallel(n_jobs=6,verbose=10,backend="multiprocessing")(delayed(measureDiffGalPhotometry)(entry,destdir="diffgal") for entry in sample)

def test(sample):

    # entry = sample[sample["survey_id"] == 8647474690858025275][0]
    entry = sample[sample["survey_id"] == 8647474690313946727][0]
    measureDiffGalPhotometry(entry, destdir="diffgal", debug=True)

def mkDiffGalCatalog(sample,destdir="diffgal"):

    from utils import getTotalNClumps
    N = getTotalNClumps(sample=sample, destdir=destdir, suffix="diffgal-phot")
    phot = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-diffgal-phot.fits".format(sample[0], destdir))

    catalog = np.recarray(N,dtype=[("GAL_ID",int)]+phot.dtype.descr)

    i = 0
    for j, entry in enumerate(sample):

        try:
            phot = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-diffgal-phot.fits".format(entry, destdir))
        except OSError:
            continue

        catalog[i:i+len(phot)]["GAL_ID"] = entry["survey_id"]
        for x in catalog.dtype.names[1:]: catalog[i:i+len(phot)][x] = phot[x]

        i += len(phot)

    fitsio.writeto("photom/{:s}/diffgal_photom.fits".format(destdir),catalog,overwrite=True)

if __name__ == "__main__":

    sample = fitsio.getdata("samples/final_clumpy_sample.fits")

    keepIDs = [8647474690313946727,8647474690858025275,
               8647474691403874778,8647475119817098026,
               8647475120909451490,8647475121445077068]
    cond = np.in1d(sample["survey_id"],keepIDs)
    sample = sample[cond]

    main(sample=sample)
    # test(sample=sample)
    mkDiffGalCatalog(sample=sample)

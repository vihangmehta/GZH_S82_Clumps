from useful import *
from utils_clumps import *
from plotter import plotClumpsCuration, InteractiveClumpCurator

from astropy.convolution import Box2DKernel,convolve,convolve_fft

def setupCatalog(sexcat,entry):

    dtype = [("ID", int),("X", float),("Y", float),("RA", float),("DEC", float),
             ("GAL_XC", float),("GAL_YC", float),("GAL_RA", float),("GAL_DEC", float),
             ("GAL_REFF", float),("GAL_SMA", float),("GAL_SMB", float),("GAL_THETA", float),
             ("GAL_REFF_XY", float),("GAL_SMA_XY", float),("GAL_SMB_XY", float),
             ("DISTANCE_XY", float),("DISTANCE", float),("DISTNORM", float),("DIST_SMA", float),("DISTPHYS", float),
             ("PSF_WIDTH_AVG", float),("PROX_FLAG", int)]

    catalog = np.recarray(len(sexcat),dtype=dtype)
    for x in catalog.dtype.names: catalog[x] = -99

    catalog["ID"]  = sexcat["NUMBER"]
    catalog["X"]   = sexcat["X_IMAGE"]
    catalog["Y"]   = sexcat["Y_IMAGE"]
    catalog["RA"]  = sexcat["X_WORLD"]
    catalog["DEC"] = sexcat["Y_WORLD"]
    catalog["PSF_WIDTH_AVG"] = entry["PSF_WIDTH_AVG"]

    return catalog

def mkSmoothSegMap(entry, filt="r", smth=5, verbose=False):

    img = "S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry, filt)
    smoothed = "photom/smooth/objID{0[survey_id]:020d}-{1:s}.smooth.fits".format(entry, filt)
    img, hdr = fitsio.getdata(img, header=True)

    ######################################################
    size = img.shape[0]
    __img = img[int(np.floor(size * 1.0 / 4.0)) : int(np.ceil(size * 3.0 / 4.0)),
                int(np.floor(size * 1.0 / 4.0)) : int(np.ceil(size * 3.0 / 4.0))]

    med, sig = np.median(__img), np.std(__img)
    img = np.clip(img, med - 50 * sig, med + 50 * sig)

    # med, sig = np.median(img), np.std(img)
    # _img = img[(med - 3 * sig < img) & (img < med + 3 * sig)]
    ######################################################

    kernel = Box2DKernel(smth)
    simg = convolve_fft(img, kernel, fill_value=np.NaN)  # np.median(_img))
    fitsio.writeto(smoothed, data=simg, header=hdr, overwrite=True)

    if   entry["survey_id"] in bristar: detectThresh = 25.00
    elif entry["survey_id"] in blended: detectThresh = 25.00
    else: detectThresh = 10.00

    args = {"det_img": "photom/smooth/objID{0[survey_id]:020d}-{1:s}.smooth.fits".format(entry,filt),
            "catname": "photom/smooth/objID{0[survey_id]:020d}-cat.smooth.fits".format(entry),
            "seg_img": "photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry),
            "detectThresh": detectThresh,
            "analysisThresh": detectThresh+0.01}

    call = "sex {det_img:s} " \
           "-c config/config_smooth.sex " \
           "-PARAMETERS_NAME config/param_smooth.sex " \
           "-CATALOG_NAME {catname:s} -CATALOG_TYPE FITS_1.0 " \
           "-WEIGHT_TYPE NONE " \
           "-DETECT_THRESH {detectThresh:.2f} -ANALYSIS_THRESH {analysisThresh:.2f} " \
           "-CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {seg_img:s} ".format(**args)

    runBashCommand(call, cwd=cwd, verbose=verbose)

def removeOutsideClumps(catalog, entry):
    """
    Remove clumps that are outside the parent galaxies' segmap
    """
    segimg = fitsio.getdata("photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry))
    segidx = segimg[int(np.round(segimg.shape[1] / 2)), int(np.round(segimg.shape[0] / 2))]

    if segidx != 0:
        cond = (segimg[np.round(catalog["Y"]-1).astype(int),
                       np.round(catalog["X"]-1).astype(int)]==segidx)
        catalog = catalog[cond]
    else:
        print("Segmap is zero at the center for ID#{0[survey_id]:020d}".format(entry))
        catalog = catalog[np.zeros(len(catalog), dtype=bool)]

    apersize = getClumpApertureSize(entry)
    stmask = getStarMask(entry,imshape=segimg.shape,radius=0.5*apersize["star"])
    cond = (stmask[np.round(catalog["Y"]-1).astype(int),
                   np.round(catalog["X"]-1).astype(int)]==0)
    catalog = catalog[cond]

    return catalog

def getCoM(catalog, entry):
    """
    Calculate the CoM of the image after masking the detected clumps
    Also removes the CoM from the clump list, if it is marked as one
    """
    ### Get smoothed image, segmap and custom clump mask
    smth_img, smth_hdr = fitsio.getdata("photom/smooth/objID{0[survey_id]:020d}-r.smooth.fits".format(entry),header=True)
    smth_cat = fitsio.getdata("photom/smooth/objID{0[survey_id]:020d}-cat.smooth.fits".format(entry))
    smth_seg = fitsio.getdata("photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry))
    segidx = smth_seg[int(np.round(smth_seg.shape[1] / 2)), int(np.round(smth_seg.shape[0] / 2))]

    ### Custom mask for stars
    apersize = getClumpApertureSize(entry)
    stmask = getStarMask(entry,imshape=smth_img.shape,radius=0.5*apersize["star"])

    _catalog = catalog.copy()
    iterate,xc_old,yc_old = True,1e10,1e10
    while iterate:

        ### Get the mask for all detected clumps
        clmask = getClumpMask(_catalog,imshape=smth_img.shape,radius=0.5*apersize["mask"])
        mask = stmask | clmask

        ### Mask the clumps as well as everything that's not the galaxy in the smoothed image
        smth_img_masked = np.ma.masked_array(smth_img, mask=(smth_seg != segidx) | mask)

        ### Calculate the CoM
        xc_new, yc_new = calcCoM(img=smth_img_masked)

        ### Remove the clumps near the new CoM
        _catalog = catalog[np.sqrt((catalog["X"]-xc_new)**2 +
                                   (catalog["Y"]-yc_new)**2) > catalog["PSF_WIDTH_AVG"]/FITS_pixscale]

        ### Compute shift in CoM
        deltaCoM = np.sqrt((xc_new-xc_old)**2 + (yc_new-yc_old)**2)
        xc_old, yc_old = xc_new, yc_new
        iterate = True if deltaCoM>1 else False     # Iterate until CoM converges to within 3 px

    ### Record the CoM
    catalog["GAL_XC"], catalog["GAL_YC"] = xc_new, yc_new

    ### Convert the center (x,y) to (ra,dec)
    wcs = WCS(smth_hdr, fix=False)
    catalog["GAL_RA"], catalog["GAL_DEC"] = wcs.all_pix2world(catalog["GAL_XC"], catalog["GAL_YC"], 1)

    ### Calculate the distance from center
    catalog["DISTANCE_XY"] = np.sqrt((catalog["X"]-catalog["GAL_XC"])**2 +
                                     (catalog["Y"]-catalog["GAL_YC"])**2)
    catalog["DISTANCE"] = catalog["DISTANCE_XY"] * FITS_pixscale

    if np.isfinite(entry["REDSHIFT_DR14"]):
        catalog["DISTPHYS"] = catalog["DISTANCE"] / Planck15.arcsec_per_kpc_proper(entry["REDSHIFT_DR14"]).value
    else:
        catalog["DISTPHYS"] = np.NaN

    ### Don't count the CoM as a clump (presumably bulge)
    catalog = catalog[catalog["DISTANCE"] > catalog["PSF_WIDTH_AVG"]]

    return catalog

def getMorphologyParameters(catalog,entry,filt="r"):

    ### Get smoothed image, segmap and custom clump mask
    img = fitsio.getdata("S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt))
    segimg = fitsio.getdata("photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry))
    segidx = segimg[int(np.round(segimg.shape[1] / 2)), int(np.round(segimg.shape[0] / 2))]

    ### Get the mask for all detected clumps
    apersize = getClumpApertureSize(entry)
    mask = getStarMask(entry,imshape=img.shape,radius=0.5*apersize["star"]) | \
           getClumpMask(catalog,imshape=img.shape,radius=0.5*apersize["mask"])

    ### Mask the clumps as well as everything that's not the galaxy in the smoothed image
    img = np.ma.masked_array(img, mask=(segimg != segidx) | mask)

    catalog["GAL_REFF_XY"] = calcReff(xc=catalog["GAL_XC"][0],yc=catalog["GAL_YC"][0],img=img.filled(-99),clumpMask=mask)
    catalog["GAL_SMA_XY"], \
    catalog["GAL_SMB_XY"], \
    catalog["GAL_THETA"] = calcEllipseParameters(xc=catalog["GAL_XC"][0],yc=catalog["GAL_YC"][0],img=img)

    catalog["GAL_REFF"] = catalog["GAL_REFF_XY"] * FITS_pixscale
    catalog["GAL_SMA"]  = catalog["GAL_SMA_XY"]  * FITS_pixscale
    catalog["GAL_SMB"]  = catalog["GAL_SMB_XY"]  * FITS_pixscale
    catalog["DISTNORM"] = catalog["DISTANCE"] / catalog["GAL_REFF"]
    catalog["DIST_SMA"] = catalog["DISTANCE"] / catalog["GAL_SMA"]

    return catalog

def getProximityFlag(catalog,entry):

    apersize = getClumpApertureSize(entry)
    pos = np.array([catalog["X"], catalog["Y"]]).T
    separation = scipy.spatial.distance.cdist(pos, pos)
    np.fill_diagonal(separation, 1e10)
    catalog["PROX_FLAG"] = np.min(separation, axis=0) <= apersize["mask"] / FITS_pixscale
    return catalog

def markClumpsToRemove(catalog,entry,destdir):

    fig,axes,transform = plotClumpsCuration(entry=entry,destdir=destdir,trimcat=catalog,markPts=False,savefig=False)
    fullcat = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-cat.fits".format(entry, destdir))
    interactiveCurator = InteractiveClumpCurator(axes=axes,transform=transform,fullcat=fullcat,trimcat=catalog)
    plt.show(block=True)

    if os.path.isfile("samples/custom_clump_rejects.fits"):
        fullRejectList = fitsio.getdata("samples/custom_clump_rejects.fits")
    else:
        fullRejectList = np.recarray(0,dtype=[("survey_id",">i8"),("X",">f8"),("Y",">f8")])

    idx = np.where(fullRejectList["survey_id"]==entry["survey_id"])[0]
    fullRejectList = np.delete(fullRejectList,idx)

    _rejectList = interactiveCurator.rejectList
    rejectList = np.recarray(_rejectList.shape[1],dtype=[("survey_id",">i8"),("X",">f8"),("Y",">f8")])
    rejectList["survey_id"] = entry["survey_id"]
    rejectList["X"] = _rejectList[0,:]
    rejectList["Y"] = _rejectList[1,:]

    fullRejectList = rfn.stack_arrays([fullRejectList,rejectList],usemask=False,asrecarray=True)
    fitsio.writeto("samples/custom_clump_rejects.fits",fullRejectList,overwrite=True)

def removeMarkedClumps(catalog,entry):

    fullRejectList = fitsio.getdata("samples/custom_clump_rejects.fits")
    rejectList = fullRejectList[fullRejectList["survey_id"]==entry["survey_id"]]
    if len(rejectList)>0:
        rejCond = np.array([np.any(
                    np.sqrt((rejectList["X"]-xc)**2+(rejectList["Y"]-yc)**2)<entry["PSF_WIDTH_AVG"]/FITS_pixscale)
                                for xc,yc in zip(catalog["X"],catalog["Y"])],dtype=bool)
        catalog = catalog[~rejCond]
    return catalog

def curateClumps(entry,destdir,interactive=False):

    catname  = "photom/{1:s}/objID{0[survey_id]:020d}-cat.fits".format(entry, destdir)
    savename = "photom/{1:s}/objID{0[survey_id]:020d}-trim.fits".format(entry, destdir)

    catalog = fitsio.getdata(catname)
    catalog = setupCatalog(catalog,entry=entry)
    catalog = removeOutsideClumps(catalog,entry=entry)

    if len(catalog)>0:
        catalog = getCoM(catalog,entry=entry)

    if len(catalog)>0:
        catalog = getMorphologyParameters(catalog,entry=entry)

    if interactive:
        markClumpsToRemove(catalog,entry=entry,destdir=destdir)

    if len(catalog)>0:
        catalog = removeMarkedClumps(catalog,entry=entry)
        # catalog = catalog[catalog["DISTNORM"]<=10]

    if len(catalog)>0:
        catalog = getProximityFlag(catalog,entry=entry)

    if len(catalog)>0:
        fitsio.writeto(savename, catalog, overwrite=True)
    else:
        print("No clumps for objID{0[survey_id]:020d}".format(entry))
        if os.path.isfile(savename): os.remove(savename)
        os.system("touch {:s}".format(savename))

    plotClumpsCuration(entry=entry,destdir=destdir)

def main(sample,destdir,interactive=False):

    Parallel(n_jobs=15,verbose=10,backend="multiprocessing")(delayed(mkSmoothSegMap)(entry) for entry in sample)

    if interactive:
        for j,entry in enumerate(sample):
            print("Curating clumps for objID#{:020d} [{:d}/{:d}] ...".format(entry["survey_id"],j+1,len(sample)))
            curateClumps(entry=entry,destdir=destdir,interactive=interactive)
    else:
        Parallel(n_jobs=15,verbose=10,backend="multiprocessing")(delayed(
            curateClumps)(entry,destdir=destdir,interactive=interactive) for entry in sample)

def test(sample,destdir,interactive=False):

    # for objid in bristar:
    #     entry = sample[sample["survey_id"] == objid][0]
    #     mkSmoothSegMap(entry,filt="r",smth=5)
    #     curateClumps(entry, destdir=destdir, interactive=interactive)

    entry = sample[sample["survey_id"] == 8647474690858025275][0]
    fig,[ax1,ax2,ax3,ax4],transform = plotClumpsCuration(entry=entry,destdir=destdir,savefig=False)

    dx = ax1.set_xlim()[1]
    xlim = [dx*0.24, dx*0.76]
    ax1.set_xlim(xlim)
    ax1.set_ylim(xlim[::-1])

    fig.savefig("plots/final/poster_curate.png")

if __name__ == "__main__":

    sample = fitsio.getdata("samples/final_clumpy_sample.fits")

    # main(sample=sample,destdir="boxcar10",interactive=False)
    test(sample=sample,destdir="boxcar10",interactive=False)

    # plt.show()

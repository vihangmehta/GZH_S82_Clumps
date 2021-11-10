from useful import *
from utils import *
from utils_DR14 import *
from utils_S82 import *
from plotter import plotSDSSSummary

def getNearbyObjectRedshift(entry,nearbyObjects,verbose=False):

    ### Look for nearby objects and sort them by distance and also only consider the objects within 5" diameter
    m1,m2,dist = matchRADec(entry["ra"],entry["dec"],nearbyObjects["ra"],nearbyObjects["dec"],crit=2.5,maxmatch=0)
    nearbyObjects = nearbyObjects[m2]
    isort = np.argsort(dist)

    ### Look through the nearby objects and see if one has a redshift
    for nearbyObject,_dist in zip(nearbyObjects[isort],dist[isort]):
        info = doObjectSearch(objId=nearbyObject["objid"])
        info = parseSkyServerQuickLookOutput(info)
        if info["objectInfo"]["specObjId"]!="":
            _z = info["SpectralData"]["redshift_z"]
            if np.isfinite(_z) and (_z>0):
                nearbyObject_z = _z
                if verbose:
                    print("Found a redshift for DR7ObjId#{0[survey_id]:d} from DR14ObjId${1:d} ({2:.3f}\" away): {3:.3f}".format(entry,nearbyObject["objid"],_dist,nearbyObject_z))
                return nearbyObject_z

    return np.NaN

def getDR14ObjSearch(entry,verbose=False):

    if verbose:
        sys.stdout.write("\rProcessing obj#{:020d} -- {:d}/{:d} ... ".format(entry["survey_id"],0,0))
        sys.stdout.flush()

    ### Setup the region size to search for nearby objects in
    ### This is currently set to use 15 kpc if there is a redshift available for the galaxy (or nearby objects)
    ### Otherwise it defaults to 20" around the main object
    angSize = min(getAngularSize(z=entry["REDSHIFT"],physSize=15,angSize=10),60)
    ### Perform a radial search with SDSS DR14
    nearbyObjects = doRadialSearch(ra=entry["ra"],dec=entry["dec"],radius=angSize/60.)

    ### If there is no redshift available for the galaxy, see if there is one available for a nearby object
    if not np.isfinite(entry["REDSHIFT"]) and len(nearbyObjects)>0:
        ### Get redshift from nearby objects if no redshift available
        nearbyObject_z = getNearbyObjectRedshift(entry=entry,nearbyObjects=nearbyObjects,verbose=verbose)
        ### With the new redshift (or just the default NaN) search again
        angSize = min(getAngularSize(z=nearbyObject_z,physSize=15,angSize=10),60)
        nearbyObjects = doRadialSearch(ra=entry["ra"], dec=entry["dec"], radius=angSize/60.)

    ### Convert from Pandas DF and setup for extra metadata
    nearbyObjects = rfn.drop_fields(nearbyObjects,'index')
    nearbyObjects = rfn.append_fields(nearbyObjects,
                                        names=["petrorad_r","extinction_r",
                                               "specobjid","run2d",
                                               "plate","mjd","fiberid",
                                               "primary","otherspec","fromspecsearch",
                                               "redshift","redshift_err","distance"],
                                        data=np.zeros((13,len(nearbyObjects)))*np.NaN,
                                        dtypes=[float]*2+["<U20","<U10"]+[int]*6+[float]*3,
                                        usemask=False,asrecarray=True)
    nearbyObjects["specobjid"] = [""]*len(nearbyObjects)

    ### Sort nearby objects by distance
    m1,m2,dist = matchRADec(entry["ra"],entry["dec"],nearbyObjects["ra"],nearbyObjects["dec"],crit=angSize*5,maxmatch=0)
    nearbyObjects = nearbyObjects[m2]
    isort = np.argsort(dist)
    nearbyObjects, dist = nearbyObjects[isort], dist[isort]

    ### For each nearby object, get the DR14 quick look metadata
    for j,nearbyObject in enumerate(nearbyObjects):

        if verbose:
            sys.stdout.write("\rProcessing obj#{:020d} -- {:d}/{:d} ... ".format(entry["survey_id"],j+1,len(nearbyObjects)))
            sys.stdout.flush()

        nearbyObject["distance"] = dist[j] * 3600

        info = doObjectSearch(objId=nearbyObject["objid"])
        info = parseSkyServerQuickLookOutput(info)

        nearbyObject["petrorad_r"] = float(info["ImagingData"]["petrorad_r"].split("&plusmn")[0])
        nearbyObject["extinction_r"] = info["ImagingData"]["extinction_r"]

        if info["objectInfo"]["specObjId"]!="":
            nearbyObject["specobjid"]      = info["objectInfo"]["specObjId"]
            nearbyObject["plate"]          = info["SpectralData"]["plate"]
            nearbyObject["mjd"]            = info["SpectralData"]["mjd"]
            nearbyObject["fiberid"]        = info["SpectralData"]["fiberid"]
            nearbyObject["run2d"]          = info["SpectralData"]["run2d"]
            nearbyObject["redshift"]       = info["SpectralData"]["redshift_z"]
            nearbyObject["redshift_err"]   = info["SpectralData"]["redshift_err"]
            nearbyObject["primary"]        = info["SpectralData"]["primary"]
            nearbyObject["otherspec"]      = info["SpectralData"]["otherspec"]
            nearbyObject["fromspecsearch"] = 0

        else:
            nearbySpecObjects = querySpecSearch(ra=nearbyObject["ra"],dec=nearbyObject["dec"],radius=2)[0]
            cond = (nearbySpecObjects["primary"]==1)
            if   sum(cond)==1:
                idx = np.where(cond)[0][0]
            elif sum(cond) >1:
                _ = np.array(nearbySpecObjects["specobjid"][cond]).astype(np.uint64)
                idx = np.where(cond)[0][np.argmax(_)]
            else:
                idx = None
            if idx is not None:
                nearbyObject["specobjid"]      = nearbySpecObjects[idx]["specobjid"]
                nearbyObject["plate"]          = nearbySpecObjects[idx]["plate"]
                nearbyObject["mjd"]            = nearbySpecObjects[idx]["mjd"]
                nearbyObject["fiberid"]        = nearbySpecObjects[idx]["fiberid"]
                nearbyObject["run2d"]          = nearbySpecObjects[idx]["run2d"]
                nearbyObject["redshift"]       = nearbySpecObjects[idx]["redshift"]
                nearbyObject["redshift_err"]   = nearbySpecObjects[idx]["redshift_err"]
                nearbyObject["primary"]        = nearbySpecObjects[idx]["primary"]
                nearbyObject["otherspec"]      = nearbySpecObjects[idx]["otherspec"]
                nearbyObject["fromspecsearch"] = 1

    if verbose:
        sys.stdout.write("done!\n")
        sys.stdout.flush()

    ### Get the spectrum for each object with specid
    for nearbyObject in nearbyObjects:
        if nearbyObject["specobjid"]!="":
            if not os.path.isfile("DR14/spectra/specID{:s}.fits".format(nearbyObject["specobjid"].zfill(20))):
                getDR14Spectrum(entry=nearbyObject,verbose=False)

    if len(nearbyObjects)>0:
        fitsio.writeto("DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"]),nearbyObjects,overwrite=True)


    #######################
    ### Spectral search ###
    #######################
    ### Search the spec_search results for nearby spectra
    nearbySpecObjects,dist = querySpecSearch(ra=entry["ra"],dec=entry["dec"],radius=angSize)

    for nearbySpecObject in nearbySpecObjects:
        if not os.path.isfile("DR14/spectra/specID{:s}.fits".format(nearbySpecObject["specobjid"].zfill(20))):
            getDR14Spectrum(entry=nearbySpecObject,verbose=False)

    if len(nearbySpecObjects)>0:
        fitsio.writeto("DR14/specObj/nearby_specObj_ID{:020d}.fits".format(entry["survey_id"]),nearbySpecObjects,overwrite=True)

def getStamps(entry,overwrite=False):

    cutoutSize = getCutoutSize(entry)

    if (not os.path.isfile("DR14/jpegs/objID{0[survey_id]:020d}.jpg".format(entry))) or \
       (not os.path.isfile("DR14/jpegs/objID{0[survey_id]:020d}_raw.jpg".format(entry))) or overwrite:
        getDR14JPEGStamp(entry=entry,angSize=cutoutSize)

    if not os.path.isfile("S82/jpegs/objID{0[survey_id]:020d}.jpg".format(entry)) or overwrite:
        getS82JPEGStamp(entry=entry)

    for filt in sdss_filters:
        if not os.path.isfile("S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt)) or overwrite:
            getS82FitsStamp(entry=entry,filt=filt,angSize=cutoutSize,overwriteFrame=False)

def main(catalog):

    # ### Get all nearby objects
    Parallel(n_jobs=10,verbose=10,backend="multiprocessing")(delayed(getDR14ObjSearch)(entry) for entry in catalog)
    mkSummaryPhotObj(catalog=catalog)
    mkSummarySpecObj(catalog=catalog)

    ### Get all FITS and JPEG stamps
    # Parallel(n_jobs=5,verbose=10,backend="multiprocessing")(delayed(getStamps)(entry) for entry in catalog)
    for j,entry in enumerate(catalog):
        print("\rCreating stamps for ObjID#{:020d} [{:3d}/{:3d}] ... ".format(entry["survey_id"],j+1,len(catalog)),end="")
        getStamps(entry,overwrite=False)
    print("done.")

    ### Make all the plots
    Parallel(n_jobs=10,verbose=10,backend="multiprocessing")(delayed(plotSDSSSummary)(entry) for entry in catalog)
    filelist = " ".join(["plots/summary_sdss/objID{0[survey_id]:020d}.png".format(entry) for entry in catalog])
    savename = "plots/pdf/gzh_sdss_clumpy_summarySDSS.pdf"
    os.system("convert {0:s} {1:s}".format(filelist,savename))

def test(catalog):

    entry = catalog[catalog["survey_id"]==8647474690883846156][0]

    # getDR14ObjSearch(entry=entry)
    # getStamps(entry=entry,overwrite=False)
    plotSDSSSummary(entry=entry,savefig=False)

if __name__ == '__main__':

    catalog = fitsio.getdata("samples/gzh_sdss_clumpy.fits")

    # main(catalog)
    test(catalog)

    plt.show()

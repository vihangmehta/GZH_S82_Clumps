from useful import *

import urllib
import SciServer
from SciServer import Authentication, CasJobs, SkyServer

Authentication_loginName = 'vihangmehta'
Authentication_loginPassword = 'highzgal'
token = Authentication.login(Authentication_loginName, Authentication_loginPassword)

def parseSkyServerQuickLookOutput(obj):

    res = {}
    for entry in obj:
        _entry = dict(entry)
        if len(_entry["Rows"])>0:
            res[_entry["TableName"]] = dict(_entry["Rows"][0])
        else:
            res[_entry["TableName"]] = {}
    return res

def doObjectSearch(objId=None,specObjId=None,dataRelease="DR14"):

    dataRelease = "DR14"

    if objId is None and specObjId is None:
        raise Exception("Please provide either an objectID or a specObjectID for objectSearch().")

    res,retry = None,0
    while (res is None):
        try:
            res = SkyServer.objectSearch(objId=objId,specObjId=specObjId,dataRelease=dataRelease)
        except requests.exceptions.ConnectionError:
            res = None
            retry += 1
            time.sleep(1)
    return res

def doRadialSearch(ra,dec,radius,dataRelease="DR14"):

    res,retry = None,0
    while (res is None):
        try:
            res = SkyServer.radialSearch(ra=ra, dec=dec, radius=radius, limit="250", dataRelease=dataRelease).to_records()
        except requests.exceptions.ConnectionError:
            res = None
            retry += 1
            time.sleep(1)
    return res

def getDR14Spectrum(entry,verbose=False):

    savename = "DR14/spectra/specID{:s}.fits".format(entry["specobjid"].zfill(20))
    try:
        url = "https://data.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/{0[plate]:04d}/spec-{0[plate]:04d}-{0[mjd]:05d}-{0[fiberid]:04d}.fits".format(entry)
        urllib.request.urlretrieve(url, savename)
        if verbose: print("Retrieved eBOSS spectrum for {:s}".format(savename))
    except urllib.error.HTTPError:
        try:
            url = "https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/spectra/{0[plate]:04d}/spec-{0[plate]:04d}-{0[mjd]:05d}-{0[fiberid]:04d}.fits".format(entry)
            urllib.request.urlretrieve(url, savename)
            if verbose: print("Retrieved SDSS spectrum for {:s}".format(savename))
        except urllib.error.HTTPError:
            print("No spectrum found for specID#{:s}".format(entry["specobjid"].zfill(20)))
            pass
    return savename

def getDR14JPEGStamp(entry,angSize,dataRelease="DR14"):

    Npix  = int(np.ceil(angSize / JPEG_pixscale))

    try: ra, dec = entry["ra"],entry["dec"]
    except ValueError: ra, dec = entry["RA"],entry["DEC"]

    try:
        img = SkyServer.getJpegImgCutout(ra=ra,dec=dec,
                                         width=Npix, height=Npix, scale=JPEG_pixscale,
                                         dataRelease=dataRelease,opt="PS")
        raw = SkyServer.getJpegImgCutout(ra=ra,dec=dec,
                                         width=Npix, height=Npix, scale=JPEG_pixscale,
                                         dataRelease=dataRelease)
    except:
        img = SkyServer.getJpegImgCutout(ra=ra,dec=dec,
                                         width=Npix, height=Npix, scale=JPEG_pixscale,
                                         dataRelease="DR13",opt="PS")
        raw = SkyServer.getJpegImgCutout(ra=ra,dec=dec,
                                         width=Npix, height=Npix, scale=JPEG_pixscale,
                                         dataRelease="DR13")

    savename = "DR14/jpegs/objID{0[survey_id]:020d}_raw.jpg".format(entry)
    rgb = Image.fromarray(raw)
    rgb.save(savename)

    savename = "DR14/jpegs/objID{0[survey_id]:020d}.jpg".format(entry)
    rgb = Image.fromarray(img)
    rgb.save(savename)
    return savename

def getSpectraSearchCatalog():

    removeIDs = ["1689976647551313920"]
    specSearch = fitsio.getdata("DR14/specSearch/gzh_sdss_clumpy_specSearch.fits")
    idx = np.where(np.in1d(specSearch["specobjid"],removeIDs))[0]
    specSearch = np.delete(specSearch,idx)
    return specSearch

specSearch = getSpectraSearchCatalog()

def getDR14SpectraMatches(ra, dec):

    m1, m2, d12 = matchRADec(ra, dec, specSearch["ra"], specSearch["dec"], crit=1.5, maxmatch=0)
    specMatch = specSearch[m2]

    cond = specMatch["primary"] == 1
    if sum(cond) == 1:
        # When there is only one primary, pick it
        idx = np.where(cond)[0][0]
    elif sum(cond) > 1:
        # When there is more than one primary, pick the one with the largest specobjid (aka latest?)
        _sid = np.array(specMatch["specobjid"][cond]).astype(np.uint64)
        idx = np.where(cond)[0][np.argmax(_sid)]
    elif len(specMatch) > 0:
        # When there are no primaries, pick the one with the largest specobjid (aka latest?)
        _sid = np.array(specMatch["specobjid"]).astype(np.uint64)
        idx = np.argmax(_sid)
    else:
        # When there are no matches, :(
        idx = -1

    specMatch = specMatch[idx] if idx >= 0 else None
    return specMatch

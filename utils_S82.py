from useful import *
from utils import *

import urllib

def getS82ImagePars(entry,filt=None):

    args = dict([(x,entry[x.upper()]) for x in ["run","rerun","camcol","field","obj"]])

    if   args["run"]==106: args["run"] = 100006
    elif args["run"]==206: args["run"] = 200006

    if filt: args["filt"] = filt
    return args

def getS82JPEGStamp(entry):

    img = Image.open("/data/extragal/willett/gzh/jpg/{0[survey_id]:d}.jpg".format(entry))
    img = np.rot90(img,1)
    img = Image.fromarray(img)
    img.save("S82/jpegs/objID{0[survey_id]:020d}.jpg".format(entry))

def getS82FitsStamp(entry,filt,angSize,overwriteFrame=False,verbose=False):

    args = getS82ImagePars(entry=entry,filt=filt)

    fitsname = "S82/frames/fpC-{run:06d}-{filt:s}{camcol:d}-{field:04d}.fits".format(**args)
    if overwriteFrame: getS82Frame(entry=entry,filt=filt,verbose=verbose)

    file_readable, retry = False, 0
    while not file_readable:
        try:
            img,img_hdr = fitsio.getdata(fitsname,0,header=True)
            file_readable = True
        except (OSError,EOFError,TypeError):
            time.sleep(2)
            getS82Frame(entry=entry,filt=filt,verbose=verbose)
            retry += 1

    savename = "S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt)
    extractSwarpCutout(img=fitsname,ra=entry["ra"],dec=entry["dec"],size=angSize,savename=savename,overwrite=overwrite,verbose=verbose)
    fixS82FitsWcs(img=savename)

def fixS82FitsWcs(img):

    oldhdu = fitsio.open(img)

    if not ("DEC" in oldhdu[0].header["CTYPE1"] and \
             "RA" in oldhdu[0].header["CTYPE2"]):
        print("{:s} doesn't seem to need its WCS fixed".format(img))
        return

    newhdu = copy.deepcopy(oldhdu)
    newhdu[0].data = oldhdu[0].data.T

    for x in ["CTYPE","CRVAL","CRPIX"]:
        newhdu[0].header[x+"1"] = oldhdu[0].header[x+"2"]
        newhdu[0].header[x+"2"] = oldhdu[0].header[x+"1"]
    newhdu[0].header["CD1_1"] = oldhdu[0].header["CD2_2"]
    newhdu[0].header["CD2_2"] = oldhdu[0].header["CD1_1"]

    newhdu.writeto(img,overwrite=True)

def getS82Frame(entry,filt,verbose=False):

    args = getS82ImagePars(entry=entry,filt=filt)

    try:
        savename = "S82/frames/fpC-{run:06d}-{filt:s}{camcol:d}-{field:04d}.fits.gz".format(**args)
        url = "http://das.sdss.org/imaging/{run:d}/{rerun:d}/corr/{camcol:d}/fpC-{run:06d}-{filt:s}{camcol:d}-{field:04d}.fit.gz".format(**args)
        urllib.request.urlretrieve(url, savename)
        if verbose: print("Retrieved {:s}-band frame: {:s}".format(filt,savename))
        runBashCommand("gunzip -f {:s}".format(savename),cwd=cwd,verbose=False)
    except urllib.error.HTTPError:
        print("No {filt:s}-band frame found for {run:06d}--{camcol:d}-{rerun:d}-{field:04d}".format(**args))

def getS82TsField(entry,overwrite=False,verbose=False):

    args = getS82ImagePars(entry=entry)

    savename = "S82/tsField/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fits".format(**args)
    if not overwrite: return savename

    try:
        url = "http://das.sdss.org/imaging/{run:d}/{rerun:d}/calibChunks/{camcol:d}/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fit".format(**args)
        urllib.request.urlretrieve(url, savename)
        if verbose: print("Retrieved tsField: {:s}".format(filt,savename))
    except urllib.error.HTTPError:
        print("No tsField found for {run:06d}-{camcol:d}-{rerun:d}-{field:04d}".format(**args))

    return savename

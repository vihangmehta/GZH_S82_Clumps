from useful import *
from utils_S82 import getS82ImagePars

filterID = dict(zip(sdss_filters,range(5)))
bSoftPar = dict(zip(sdss_filters,[x*1e-10 for x in [1.4,0.9,1.2,1.8,7.4]]))

fixExptime = 53.907456
arcsec2PerPx2 = FITS_pixscale**2

def calcImgBackground(img,sseg):

    idx  = [int(np.floor(img.shape[1]*1/3)), int(np.ceil(img.shape[1]*2/3))]
    img  = img[ idx[0]:idx[1],idx[0]:idx[1]]
    sseg = sseg[idx[0]:idx[1],idx[0]:idx[1]]
    data = img[sseg==0]

    med, std = np.median(data),np.std(data)
    cond = (med-3*std<data) & (data<med+3*std)
    bckgd = np.std(data[cond])
    return bckgd

def skyPerArcsec2ToCounts(x,aa,kk,air,b,exptime):

    return x * exptime * 10**(-0.4 * (aa + kk*air)) * arcsec2PerPx2

def convertSky(tsF,filt,exptime):

    fID = filterID[filt]
    b   = bSoftPar[filt]
    aa  = tsF["aa"][fID]
    kk  = tsF["kk"][fID]
    air = tsF["airmass"][fID]
    sky = tsF["sky"][fID]
    err = tsF["skyerr"][fID]

    sky = skyPerArcsec2ToCounts(x=sky,aa=aa,kk=kk,air=air,b=b,exptime=exptime)
    err = skyPerArcsec2ToCounts(x=err,aa=aa,kk=kk,air=air,b=b,exptime=exptime)
    return sky, err

def calcError(flux,npix,tsF,filt,exptime):

    fID  = filterID[filt]
    gain = tsF["gain"][fID]
    dark_var = tsF["dark_variance"][fID]
    sky, skyErr = convertSky(tsF=tsF,filt=filt,exptime=exptime)
    error = np.sqrt((flux*exptime + sky*npix)/gain + npix*(dark_var + skyErr)) / exptime
    return error

def calcTsfError(entry,flux,npix,filt):

    fID  = filterID[filt]
    args = getS82ImagePars(entry,filt=filt)
    hdr = fitsio.getheader("S82/frames/fpC-{run:06d}-{filt:s}{camcol:d}-{field:04d}.fits".format(**args))
    tsF = fitsio.getdata("S82/tsField/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fits".format(**args))[0]

    flux = np.atleast_1d(flux)
    coadds = [hdr[x] for x in hdr if "COADD" in x]
    error = np.zeros((len(coadds),len(flux)))
    for k,coadd in enumerate(coadds):

        coadd = coadd.split('.')[0]
        args = {"run":    int(coadd.split('-')[1]),
                "camcol": int(coadd.split('-')[2][1]),
                "field":  int(coadd.split('-')[3])}

        args["rerun"] = 40
        savename40 = "S82/tsField.single/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fits".format(**args)
        exists40 = os.path.isfile(savename40)

        args["rerun"] = 41
        savename41 = "S82/tsField.single/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fits".format(**args)
        exists41 = os.path.isfile(savename41)

        if   exists40: tsF = fitsio.getdata(savename40)[0]
        elif exists41: tsF = fitsio.getdata(savename41)[0]

        fluxScale = calcFluxScale(zp0=30,zp1=np.abs(tsF["aa"][fID]))
        flux = flux * fluxScale / len(coadds)
        error[k,:] = calcError(flux=flux,
                               npix=npix,
                               tsF=tsF,filt=filt,exptime=fixExptime) / fluxScale

    error = np.average(error,axis=0)/np.sqrt(len(coadd))
    return error

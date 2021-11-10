from useful import *
from utils_S82 import getS82ImagePars
from utils_clumps import getClumpApertureSize, getClumpApertures, getClumpMask
from plotter import getVminVmax

filterID = dict(zip(sdss_filters,range(5)))
bSoftPar = dict(zip(sdss_filters,[x*1e-10 for x in [1.4,0.9,1.2,1.8,7.4]]))

fixExptime = 53.907456
arcsec2PerPx2 = FITS_pixscale**2

def calcFluxScale(zp0,zp1):
    """
    -2.5*log(f1) + zp1 = -2.5*log(f0) + zp0
    f1/f0 = 10**((zp1 - zp0) / 2.5)
    """
    fscale = 10**((zp1 - zp0) / 2.5)
    return fscale

def asinhToCounts(x,aa,kk,air,b,exptime):

    f_f0 = 2 * b * np.sinh(-0.4*np.log(10)*x - np.log(b))
    counts = f_f0 * exptime / 10**(0.4 * (aa + kk*air))
    return counts

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

def computeErrors(destdir="boxcar10"):

    sample = fitsio.getdata("samples/final_clumpy_sample.fits")
    sample = sample[sample["survey_id"] == 8647474690337538126]

    dtype = [("objID",int),("clumpID",int),("npix",int)]
    for filt in sdss_filters:
        dtype.extend([("flux_{:s}".format(filt),float),
                      ("ferr_{:s}".format(filt),float),
                      ("ferr1_{:s}".format(filt),float),
                      ("ferr2_{:s}".format(filt),float),
                      ("sky1_{:s}".format(filt),float),
                      ("err1_{:s}".format(filt),float),
                      ("npix_annl_{:s}".format(filt),int),
                      ("sky2_{:s}".format(filt),float),
                      ("err2_{:s}".format(filt),float)])
    stats = np.recarray(0,dtype=dtype)

    for k,entry in enumerate(sample):

        print("\rProcessing {:d}/{:d} ... ".format(k+1,len(sample)),end="",flush=True)
        catname  = "photom/{1:s}/objID{0[survey_id]:020d}-trim.fits".format(entry, destdir)
        ssegname = "photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry)

        try:
            catalog = fitsio.getdata(catname)
        except OSError:
            catalog = None

        if catalog is not None:

            _stats = np.recarray(len(catalog),dtype=dtype)
            _stats["objID"] = entry["survey_id"]
            _stats["clumpID"] = catalog["ID"]

            ssegmap = fitsio.getdata(ssegname)
            apersize = getClumpApertureSize(entry=entry)
            apertures = getClumpApertures(catalog,entry=entry)
            clump_mask = getClumpMask(catalog,imshape=ssegmap.shape,radius=0.5*apersize["mask"])
            _stats["npix"] = apertures["aper"].area

            for j,filt in enumerate(sdss_filters):

                imgname = "S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry, filt)
                img, img_hdr = fitsio.getdata(imgname, header=True)

                ### Clump Photometry
                photom = aperture_photometry(img, apertures["aper"], method="subpixel", subpixels=5)
                _stats["flux_{:s}".format(filt)] = photom["aperture_sum"]
                _stats["ferr_{:s}".format(filt)] = calcTsfError(entry=entry,flux=_stats["flux_{:s}".format(filt)],npix=_stats["npix"],filt=filt)

                ### Diffuse galaxy light calc
                mask_img = img.copy()
                mask_img[clump_mask] = -99.0
                for i, mask in enumerate(apertures["annl"].to_mask(method="center")):
                    annulus = mask.multiply(mask_img)
                    annulus = annulus[(annulus != -99.0) & (mask.data != 0)]
                    _stats["sky1_{:s}".format(filt)][i] = np.median(annulus)
                    _stats["err1_{:s}".format(filt)][i] = np.std(annulus)
                    _stats["npix_annl_{:s}".format(filt)][i] = len(annulus)

                ### Sky calc
                bckgd = img[ssegmap==0]
                med,std = np.median(bckgd),np.std(bckgd)
                bckgd = bckgd[(med-5*std<bckgd) & (bckgd<med+5*std)]
                _stats["sky2_{:s}".format(filt)] = np.median(bckgd)
                _stats["err2_{:s}".format(filt)] = np.std(bckgd)

                ### Flux calc
                _stats["flux_{:s}".format(filt)] -= _stats["sky1_{:s}".format(filt)] * _stats["npix"]
                _stats["ferr1_{:s}".format(filt)] = _stats["err1_{:s}".format(filt)] * np.sqrt(_stats["npix"])
                _stats["ferr2_{:s}".format(filt)] = _stats["err2_{:s}".format(filt)] * np.sqrt(_stats["npix"])

            stats = rfn.stack_arrays([stats,_stats],usemask=False,asrecarray=True,autoconvert=False)

    print("done.")
    fitsio.writeto("samples/sky_errors.fits",stats,overwrite=True)

def plotErrorComparison():

    oldcat = fitsio.getdata("../process/samples/final_clumps_catalog.lowz.fits")
    errors = fitsio.getdata("samples/sky_errors.fits")

    fig,axes = plt.subplots(3,2,figsize=(15,12),dpi=75)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.05,top=0.98,wspace=0.15,hspace=0.05)
    axes[-1,-1].set_visible(False)
    axes = axes.T.flatten()[:-1]

    for ax,filt in zip(axes,sdss_filters):

        ax.text(0.02,0.02,"{:s}-band".format(filt),va="bottom",ha="left",fontsize=24,fontweight=600,transform=ax.transAxes)
        ax.scatter(oldcat["clump_mag_{:s}".format(filt)],
                   oldcat["clump_flux_{:s}".format(filt)]/oldcat["clump_fluxerr_{:s}".format(filt)],c='k',s=20,lw=0,label="Old catalog")
        ax.scatter(-2.5*np.log10(errors["flux_{:s}".format(filt)])+30,
                    errors["flux_{:s}".format(filt)]/errors["err2_{:s}".format(filt)],c='tab:red',s=20,lw=0,label="from sky variation")
        ax.scatter(-2.5*np.log10(errors["flux_{:s}".format(filt)])+30,
                    errors["flux_{:s}".format(filt)]/errors["err1_{:s}".format(filt)],c='tab:blue',s=20,lw=0,label="from annulus variation")
        ax.scatter(-2.5*np.log10(errors["flux_{:s}".format(filt)])+30,
                    errors["flux_{:s}".format(filt)]/errors["ferr_{:s}".format(filt)],c='tab:green',s=20,lw=0,label="from indiv. frames")

    axes[ 2].set_xlabel("Magnitude [AB]",fontsize=18)
    axes[-1].set_xlabel("Magnitude [AB]",fontsize=18)

    leg = axes[-1].legend(loc="center",fontsize=20,handlelength=0,handletextpad=0,markerscale=0,framealpha=0,bbox_to_anchor=[0.5,-0.6])
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor()[0])
        txt.set_fontweight(600)
        hndl.set_visible(False)

    for ax in axes:
        ax.set_xlim(16.6,26.2)
        ax.set_ylim(1e-1,1e4)
        ax.set_yscale("log")
        ax.set_ylabel("SN",fontsize=18)
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]
    [tick.set_visible(False) for tick in axes[0].get_xticklabels()+axes[1].get_xticklabels()+axes[3].get_xticklabels()]

def plotErrorComparison2():

    oldcat = fitsio.getdata("../process/samples/final_clumps_catalog.lowz.fits")
    newcat = fitsio.getdata("samples/final_clump_catalog.fits")

    fig,axes = plt.subplots(3,2,figsize=(15,12),dpi=75)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.05,top=0.98,wspace=0.15,hspace=0.05)
    axes[-1,-1].set_visible(False)
    axes = axes.T.flatten()[:-1]

    for ax,filt in zip(axes,sdss_filters):

        ax.text(0.02,0.02,"{:s}-band".format(filt),va="bottom",ha="left",fontsize=24,fontweight=600,transform=ax.transAxes)
        ax.scatter(oldcat["clump_mag_{:s}".format(filt)],
                   oldcat["clump_flux_{:s}".format(filt)]/oldcat["clump_fluxerr_{:s}".format(filt)],c='k',s=20,lw=0,label="Old catalog")
        ax.scatter(newcat["clump_orgmag_{:s}".format(filt)],
                   newcat["clump_orgflux_{:s}".format(filt)]/newcat["clump_orgfluxerr_{:s}".format(filt)],c='tab:green',s=20,lw=0,label="Clump+Galaxy")
        ax.scatter(newcat["clump_undmag_{:s}".format(filt)],
                   newcat["clump_undflux_{:s}".format(filt)]/newcat["clump_undfluxerr_{:s}".format(filt)],c='tab:blue',s=20,lw=0,label="Galaxy")
        ax.scatter(newcat["clump_mag_{:s}".format(filt)],
                   newcat["clump_flux_{:s}".format(filt)]/newcat["clump_fluxerr_{:s}".format(filt)],c='orange',s=20,lw=0,label="Clump")
        ax.scatter(newcat["clump_mag_{:s}".format(filt)],
                   newcat["clump_flux_{:s}".format(filt)]/np.sqrt(newcat["clump_fluxerr_{:s}".format(filt)]**2+newcat["clump_undfluxerr_{:s}".format(filt)]**2),c='purple',s=20,lw=0,label="Clump [errfix]")

    axes[ 2].set_xlabel("Magnitude [AB]",fontsize=18)
    axes[-1].set_xlabel("Magnitude [AB]",fontsize=18)

    leg = axes[-1].legend(loc="center",fontsize=20,handlelength=0,handletextpad=0,markerscale=0,framealpha=0,bbox_to_anchor=[0.5,-0.6])
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor()[0])
        txt.set_fontweight(600)
        hndl.set_visible(False)

    for ax in axes:
        ax.set_xlim(16.6,26.2)
        ax.set_ylim(1e-1,1e4)
        ax.set_yscale("log")
        ax.set_ylabel("SN",fontsize=18)
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]
    [tick.set_visible(False) for tick in axes[0].get_xticklabels()+axes[1].get_xticklabels()+axes[3].get_xticklabels()]

def plotErrorComparisonStampsheet():

    def plotTwinStamps(objid,gs):

        gssub = gs.subgridspec(1,2,hspace=0,wspace=0)
        ax1 = fig.add_subplot(gssub[0])
        img = mpimg.imread("S82/jpegs/objID{:020d}.jpg".format(objid))
        ax1.imshow(img)

        ax2 = fig.add_subplot(gssub[1])
        img = fitsio.getdata("S82/fits/objID{:020d}-r.fits".format(objid))
        vmin,vmax = getVminVmax(img)
        ax2.imshow(img,vmin=vmin,vmax=vmax,cmap=plt.cm.Greys)

        for ax in [ax1,ax2]:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        return ax1,ax2

    catalog = fitsio.getdata("samples/final_clump_catalog.fits")
    errors  = fitsio.getdata("samples/sky_errors.fits")

    mag,sno,sna = {},{},{}
    for filt in sdss_filters:
        mag[filt] = -2.5*np.log10(errors["flux_{:s}".format(filt)])+30
        sno[filt] = errors["flux_{:s}".format(filt)] / errors["ferr_{:s}".format(filt)]
        sna[filt] = errors["flux_{:s}".format(filt)] / errors["err1_{:s}".format(filt)]

    filt = "r"
    cond_bright = (sna[filt] / sno[filt] < 1/2)
    cond_faint  = (sna[filt] / sno[filt] > 2)
    cond_normal = (1/2 < sna[filt] / sno[filt]) & (sna[filt] / sno[filt] < 2)

    print(len(np.unique(catalog["gzh_id"][cond_bright])),
          len(np.unique(catalog["gzh_id"][cond_faint])),
          len(np.unique(catalog["gzh_id"][cond_normal])),)

    sample_bright = np.random.choice(np.unique(catalog["gzh_id"][cond_bright]),25,replace=False)
    sample_faint  = np.random.choice(np.unique(catalog["gzh_id"][cond_faint ]),25,replace=False)
    sample_normal = np.random.choice(np.unique(catalog["gzh_id"][cond_normal]),25,replace=False)

    for sample,suffix in zip([sample_bright,sample_faint,sample_normal],
                             ["bright","faint","normal"]):

        fig = plt.figure(figsize=(20*1.5,10*1.5),dpi=150)
        gsgrid = fig.add_gridspec(5,5)
        fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.99,wspace=0.02,hspace=0.02)

        for gs,objid in zip(gsgrid,sample):

            ax1,ax2 = plotTwinStamps(objid=objid,gs=gs)
            idx = np.where(catalog["gzh_id"]==objid)[0]
            color = np.array(["none"]*len(idx),dtype="U9")
            color[cond_bright[idx]] = "skyblue"
            color[cond_normal[idx]] = "lawngreen"
            color[cond_faint[ idx]] = "red"
            ax2.scatter(catalog["clump_x"][idx]-1,catalog["clump_y"][idx]-1,s=30,facecolor="none",edgecolor=color,marker="o",lw=1)

        fig.savefig("plots/error_test/stamps_errorComparison_{:s}.png".format(suffix))

if __name__ == '__main__':

    # computeErrors()
    # plotErrorComparison()
    plotErrorComparison2()
    # plotErrorComparisonStampsheet()
    plt.show()

from useful import *
from utils_clumps import *

def compare_Nclumps():

    newcat = fitsio.getdata("samples/final_clump_catalog.fits")
    oldcat = fitsio.getdata("../process/samples/final_clumps_catalog.lowz.fits")

    fig,ax = plt.subplots(1,1,figsize=(10,9.5),dpi=75,tight_layout=True)

    newN,oldN = np.zeros((2,len(newcat)),dtype=int)

    uniqId = np.unique(newcat["gzh_id"],return_index=True)[1]
    for j,entry in enumerate(newcat[uniqId]):
        newN[j] = sum( newcat["gzh_id"]==entry["gzh_id"])
        oldN[j] = sum((oldcat["gzh_id"]==entry["gzh_id"]) | (oldcat["gzh_id"]==entry["gzh_id_single"]))
        if np.abs(newN[j]-oldN[j])>3:
            print(entry["gzh_id"],entry["gzh_id_single"],newN[j],oldN[j])

    newN[(oldN==0) & (newN==0)] = -99
    oldN[(oldN==0) & (newN==0)] = -99

    bins = np.arange(-0.5,30,1)
    hist = np.histogram2d(oldN,newN,bins=[bins,bins])[0]
    hist = np.ma.masked_array(hist,mask=hist==0)

    im = ax.pcolormesh(bins,bins,hist.T,cmap=plt.cm.viridis)

    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="4%", pad=0.1)
    cbax = fig.colorbar(mappable=im, cax=cbaxes, orientation="vertical")
    cbax.ax.tick_params(labelsize=15)

    lims = [-0.5,30]
    ax.plot(lims,lims,c='k',lw=1,alpha=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect(1)
    ax.set_xlabel("# Clumps (old)",fontsize=18)
    ax.set_ylabel("# Clumps (new)",fontsize=18)

    [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

def compare_photometry():

    newcat = fitsio.getdata("samples/final_clump_catalog.fits")
    oldcat = fitsio.getdata("../process/samples/final_clumps_catalog.lowz.fits")

    m1,m2,dist = matchRADec(newcat["clump_ra"],newcat["clump_dec"],oldcat["clump_ra"],oldcat["clump_dec"],crit=3,maxmatch=1)
    cond = m2!=len(oldcat)
    m1,m2 = m1[cond],m2[cond]

    newcat = newcat[m1]
    oldcat = oldcat[m2]

    fig,ax = plt.subplots(1,1,figsize=(10,9.5),dpi=75,tight_layout=True)

    ax.scatter(oldcat["clump_mag_r"],newcat["clump_mag_r"],c='k',s=15,lw=0,alpha=0.8)

    lims = [10,30]
    ax.plot(lims,lims,c='k',lw=1,alpha=0.8)
    ax.set_xlim(25,18)
    ax.set_ylim(25,18)
    ax.set_aspect(1)
    ax.set_xlabel("r-band mag (old)",fontsize=18)
    ax.set_ylabel("r-band mag (new)",fontsize=18)
    [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

def getTsField(entry,filt):

    import urllib
    from utils_S82 import getS82ImagePars

    args = getS82ImagePars(entry,filt=filt)
    fpC = "S82/frames/fpC-{run:06d}-{filt:s}{camcol:d}-{field:04d}.fits".format(**args)
    hdr = fitsio.getheader(fpC)

    for x in hdr:

        if "COADD" in x:
            tsF = hdr[x].split('.')[0]
            args = {"run": int(tsF.split('-')[1]),
                    "camcol": int(tsF.split('-')[2][1]),
                    "field": int(tsF.split('-')[3])}

            success40,success41 = -1,-1

            args["rerun"] = 40
            url = "http://das.sdss.org/imaging/{run:d}/{rerun:d}/calibChunks/{camcol:d}/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fit".format(**args)
            savename = "test_tsF/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fits".format(**args)
            if not os.path.isfile(savename):
                try:
                    urllib.request.urlretrieve(url, savename)
                    success40 = 1
                except urllib.error.HTTPError:
                    success40 = 0
            else: success40 = 1

            args["rerun"] = 41
            url = "http://das.sdss.org/imaging/{run:d}/{rerun:d}/calibChunks/{camcol:d}/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fit".format(**args)
            savename = "test_tsF/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fits".format(**args)
            if not os.path.isfile(savename):
                try:
                    urllib.request.urlretrieve(url, savename)
                    success41 = 1
                except urllib.error.HTTPError:
                    success41 = 0
            else: success41 = 1

            if (success40==1) and (success41==1):
                print("Both {run:06d}-{camcol:d}-{rerun:d}-{field:04d}".format(**args))
            if (success40==0) and (success41==0):
                print("Neither {run:06d}-{camcol:d}-{rerun:d}-{field:04d}".format(**args))

def calcTsfError(entry,catalog,filt):

    from utils_S82 import getS82ImagePars

    args = getS82ImagePars(entry,filt=filt)
    fpC = "S82/frames/fpC-{run:06d}-{filt:s}{camcol:d}-{field:04d}.fits".format(**args)
    hdr = fitsio.getheader(fpC)

    error = np.zeros((len(catalog),0))
    sky = np.zeros(0)
    for x in hdr:

        if "COADD" in x:
            tsF = hdr[x].split('.')[0]
            args = {"run":    int(tsF.split('-')[1]),
                    "camcol": int(tsF.split('-')[2][1]),
                    "field":  int(tsF.split('-')[3])}

            args["rerun"] = 40
            savename40 = "test_tsF/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fits".format(**args)
            exists40 = os.path.isfile(savename40)

            args["rerun"] = 41
            savename41 = "test_tsF/tsField-{run:06d}-{camcol:d}-{rerun:d}-{field:04d}.fits".format(**args)
            exists41 = os.path.isfile(savename41)

            if exists40 and exists41: print("both")
            elif not exists40 and not exists41: print("neither")
            elif exists40 and not exists41: tsF = fitsio.getdata(savename40)[0]
            elif not exists40 and exists41: tsF = fitsio.getdata(savename41)[0]

            idx = [filtID[filt]]
            err = np.sqrt((catalog["flux_{:s}".format(filt)]+tsF["sky"][idx]*catalog["npix"])/tsF["gain"][idx] + catalog["npix"]*(tsF["dark_variance"][idx]+tsF["skyErr"][idx]))
            error = np.column_stack((error,err))
            sky = np.append(sky,tsF["sky"][idx])

    return sky, error # np.sqrt(np.sum(error**2))

def compute_photerr(destdir="boxcar10",debug=False):

    sample = fitsio.getdata("samples/final_clumpy_sample.fits")

    dtype = [("objID",float),("clumpID",float),("npix",int)]
    for filt in sdss_filters:
        dtype.extend([("flux_{:s}".format(filt),float),
                      ("sky1_{:s}".format(filt),float),
                      ("err1_{:s}".format(filt),float),
                      ("npix_annl_{:s}".format(filt),int),
                      ("sky2_{:s}".format(filt),float),
                      ("err2_{:s}".format(filt),float)])
    stats = np.recarray(0,dtype=dtype)

    for entry in sample:

        catname  = "photom/{1:s}/objID{0[survey_id]:020d}-trim.fits".format(entry, destdir)
        detname  = "photom/{1:s}/objID{0[survey_id]:020d}-det.fits".format(entry, destdir)
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

            stats = rfn.stack_arrays([stats,_stats],usemask=False,asrecarray=True,autoconvert=False)

            for filt in sdss_filters:
                sky,err = calcTsfError(entry,catalog=_stats,filt=filt)
                print(sky,_stats["flux_{:s}".format(filt)],-2.5*np.log10(_stats["flux_{:s}".format(filt)])+30)
                plt.plot(sky,err[0,:])
                plt.show()

    fitsio.writeto("samples/sky_errors.fits",stats,overwrite=True)

def compare_photerr():

    errors = fitsio.getdata("samples/sky_errors.fits")
    uniqID = np.unique(errors["objID"],return_index=True)[1]

    fig,[[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,figsize=(18,12),dpi=75,tight_layout=True)

    colors = plt.cm.rainbow(np.linspace(0,1,len(sdss_filters)))

    for filt,color in zip(sdss_filters,colors):

        ax1.scatter(errors["err1_{:s}".format(filt)],errors["err2_{:s}".format(filt)],color=color,lw=0,s=20,alpha=0.75)
        ax2.scatter(errors["err1_{:s}".format(filt)],errors["npix_{:s}".format(filt)],color=color,lw=0,s=20,alpha=0.75)
        ax3.scatter(errors["sky1_{:s}".format(filt)],errors["err1_{:s}".format(filt)],facecolor="none",edgecolor=color,marker='o',lw=0.5,s=20,alpha=0.75)
        ax3.scatter(errors["sky2_{:s}".format(filt)],errors["err2_{:s}".format(filt)],facecolor=color,edgecolor="none",marker='o',lw=0,s=20,alpha=0.75)

    ax1.plot([-1e10,1e10],[-1e10,1e10],color='k',alpha=0.8)
    ax1.set_xlim(2e0,1.1e3)
    ax1.set_xscale("log")
    ax1.set_xlabel("$\\sigma_{sky}$ from annulus",fontsize=18)
    ax1.set_ylim(2e0,1.1e3)
    ax1.set_yscale("log")
    ax1.set_ylabel("$\\sigma_{sky}$ from image",fontsize=18)

    ax2.set_xlim(2e0,1.1e3)
    ax2.set_xscale("log")
    ax2.set_xlabel("$\\sigma_{sky}$ from annulus",fontsize=18)
    ax2.set_ylim(0,100)
    ax2.set_xlabel("Npix in annulus",fontsize=18)

    for ax in [ax3,ax4]:
        ax.set_ylim(2e0,1.1e3)
        ax.set_yscale("log")
        ax.set_ylabel("$\\sigma_{sky}$",fontsize=18)
        ax.set_xlim(-1e2,1e3)
        ax.set_xscale("symlog",linthreshx=1e0)
        ax.set_xlabel("Sky Value",fontsize=18)

    for ax in [ax1,ax2,ax3,ax4]:
        [tick.set_fontsize(15) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

if __name__ == '__main__':

    # compare_Nclumps()
    # compare_photometry()

    compute_photerr(destdir="boxcar10",debug=True)
    # compare_photerr()

    plt.show()

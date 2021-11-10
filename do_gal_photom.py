from useful import *
from utils_errors import *

from gal_extinction import Gal_Extinction
from plotter import getVminVmax,reprocessSegMap

def runSwarp(entry,verbose=False):

    det_filters = "ugri"
    imgs = dict([(filt,"S82/fits/objID{0[survey_id]:020d}-{1:1s}.fits".format(entry,filt)) for filt in det_filters])
    imshape = np.array(fitsio.getdata(imgs["r"]).shape)

    args = {"input_img": ",".join([imgs[_] for _ in det_filters]),
            "det_img": "photom/galaxy/objID{0[survey_id]:020d}-det.fits".format(entry),
            "det_wht": "photom/galaxy/objID{0[survey_id]:020d}-det.wht.fits".format(entry),
            "pixscale": FITS_pixscale,
            "img_size": ",".join(imshape.astype("U5"))}

    call = "swarp {input_img:s} -c config/config_galaxy.swarp " \
           "-WEIGHT_TYPE NONE " \
           "-IMAGEOUT_NAME {det_img:s} -WEIGHTOUT_NAME {det_wht:s} " \
           "-COMBINE Y -COMBINE_TYPE AVERAGE -RESAMPLE N -SUBTRACT_BACK N " \
           "-PIXELSCALE_TYPE MANUAL -PIXEL_SCALE {pixscale:.4f} -IMAGE_SIZE {img_size:s}".format(**args)

    runBashCommand(call, cwd=cwd, verbose=verbose)
    os.remove(args["det_wht"])

def runSextractor(entry,verbose=False):

    imgs = dict([(filt,"S82/fits/objID{0[survey_id]:020d}-{1:1s}.fits".format(entry,filt)) for filt in sdss_filters])

    for filt in sdss_filters:

        args = {"sci_img": imgs[filt],
                "det_img": "photom/galaxy/objID{0[survey_id]:020d}-det.fits".format(entry),
                "catname": "photom/galaxy/objID{0[survey_id]:020d}-{1:s}.cat.fits".format(entry,filt),
                "seg_img": "photom/galaxy/objID{0[survey_id]:020d}-seg.fits".format(entry),
                "zp": S82_ZP}

        call = "sex {det_img:s},{sci_img:s} " \
               "-c config/config_galaxy.sex " \
               "-PARAMETERS_NAME config/param_galaxy.sex " \
               "-CATALOG_NAME {catname:s} -CATALOG_TYPE FITS_1.0 -MAG_ZEROPOINT {zp:.2f} " \
               "-WEIGHT_TYPE NONE,NONE " \
               "-CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {seg_img:s}".format(**args)

        runBashCommand(call, cwd=cwd, verbose=verbose)

def removeGalacticExtinction(catalog):

    gal_ext = Gal_Extinction()
    catalog['extinct_ebv'] = gal_ext.calc_EBV(catalog['ra'],catalog['dec'])
    catalog['extinct_av'] = gal_ext.calc_Av(ebv=catalog['extinct_ebv'])

    for filt in sdss_filters:
        catalog["extinct_{:s}".format(filt)] = gal_ext.calc_Alambda(filt=filt,Av=catalog["extinct_av"])[0]
        for x in ["auto","iso"]:
            catalog[   "flux_{:s}_{:s}".format(x,filt)] = gal_ext.remove_gal_ext(flux=catalog[   "flux_{:s}_{:s}".format(x,filt)],filt=filt,Av=catalog["extinct_av"])
            catalog["fluxerr_{:s}_{:s}".format(x,filt)] = gal_ext.remove_gal_ext(flux=catalog["fluxerr_{:s}_{:s}".format(x,filt)],filt=filt,Av=catalog["extinct_av"])

    return catalog

def calcMagnitudes(catalog):

    for filt in sdss_filters:
        for x in ["auto","iso"]:
            flux    = catalog[   "flux_{:s}_{:s}".format(x,filt)]
            fluxerr = catalog["fluxerr_{:s}_{:s}".format(x,filt)]
            cond = (flux > 0)
            catalog[   "mag_{:s}_{:s}".format(x,filt)][cond] = (-2.5 * np.log10(flux[cond]) + S82_ZP)
            catalog["magerr_{:s}_{:s}".format(x,filt)][cond] = ( 2.5 / np.log(10) * (fluxerr[cond] / flux[cond]))

    return catalog

def convertFluxTouJy(catalog):

    fluxScale = calcFluxScale(zp0=23.9,zp1=S82_ZP)

    for filt in sdss_filters:
        for x in ["auto","iso"]:
            catalog[       "flux_{:s}_{:s}".format(x,filt)] /= fluxScale
            catalog[    "fluxerr_{:s}_{:s}".format(x,filt)] /= fluxScale
            catalog["fluxerr_sex_{:s}_{:s}".format(x,filt)] /= fluxScale

    return catalog

def mkGalaxyPhotom():

    catalog = fitsio.getdata('samples/final_clumpy_sample.fits')

    dtype = [("survey_id",int),("ra",float),("dec",float),("extinct_ebv",float),("extinct_av",float)]
    for filt in sdss_filters:
        for aper in ["auto","iso"]:
            dtype += [(       "flux_%s_%s"%(aper,filt),float),
                      (    "fluxerr_%s_%s"%(aper,filt),float),
                      ("fluxerr_sex_%s_%s"%(aper,filt),float),
                      (        "mag_%s_%s"%(aper,filt),float),
                      (     "magerr_%s_%s"%(aper,filt),float),
                      ( "magerr_sex_%s_%s"%(aper,filt),float)]
        dtype += [("extinct_%s"%filt,float)]
    gal_cat = np.recarray(len(catalog),dtype=dtype)

    for i,entry in enumerate(catalog):

        print("\rProcessing {:d}/{:d} ... ".format(i+1,len(catalog)),end="",flush=True)

        clump_cat = fitsio.getdata("photom/boxcar10/objID{0[survey_id]:020d}-phot.fits".format(entry))
        COMx,COMy = clump_cat["GAL_XC"][0],clump_cat["GAL_YC"][0]

        segimg = fitsio.getdata("photom/galaxy/objID{0[survey_id]:020d}-seg.fits".format(entry))
        segidx = segimg[int(np.round(COMy-1)), int(np.round(COMx-1))]

        if segidx==0:
            print(entry["survey_id"],"SEGMAP is 0 at stamp center!")
            plt.imshow(segimg,origin="lower")
            plt.scatter(COMx-1,COMy-1,color="w",lw=2,marker="x")
            plt.show()

        gal_cat["survey_id"][i] = entry["survey_id"]

        for filt in sdss_filters:

            cat = fitsio.getdata("photom/galaxy/objID{0[survey_id]:020d}-{1:s}.cat.fits".format(entry,filt))
            cat_entry = cat[segidx-1]

            if filt=="r":
                gal_cat["ra"][i]  = cat_entry["X_WORLD"]
                gal_cat["dec"][i] = cat_entry["Y_WORLD"]

            npix_auto = np.pi * (cat_entry["a_image"]*cat_entry["kron_radius"]) * (cat_entry["b_image"]*cat_entry["kron_radius"])
            gal_cat["fluxerr_auto_%s"%filt][i] = calcTsfError(entry=entry,filt=filt,flux=cat_entry["flux_auto"],npix=npix_auto)
            gal_cat["fluxerr_iso_%s"%filt][i]  = calcTsfError(entry=entry,filt=filt,flux=cat_entry["flux_iso" ],npix=cat_entry["isoarea_image"])

            for aper in ["auto","iso"]:
                gal_cat[       "flux_%s_%s"%(aper,filt)][i] = cat_entry["flux_%s"%aper]
                gal_cat["fluxerr_sex_%s_%s"%(aper,filt)][i] = cat_entry["fluxerr_%s"%aper]
                gal_cat[        "mag_%s_%s"%(aper,filt)][i] = cat_entry["mag_%s"%aper]
                gal_cat[ "magerr_sex_%s_%s"%(aper,filt)][i] = cat_entry["magerr_%s"%aper]
                gal_cat[     "magerr_%s_%s"%(aper,filt)][i] = (2.5 / np.log(10)) * (gal_cat["fluxerr_%s_%s"%(aper,filt)][i] / gal_cat["flux_%s_%s"%(aper,filt)][i])

    gal_cat = removeGalacticExtinction(gal_cat)
    gal_cat = calcMagnitudes(gal_cat)
    gal_cat = convertFluxTouJy(gal_cat)

    print("done.")
    fitsio.writeto("photom/galaxy/galaxy_photom.fits",gal_cat,overwrite=True)

def chkSegmaps():

    objlist = fitsio.getdata("samples/final_clumpy_sample.fits")["survey_id"]

    for filt in list(sdss_filters)+["seg","smthseg"]:

        print("\rPlotting {:s} ... ".format(filt),end="")
        fig,axes = plt.subplots(10,10,figsize=(20,20),dpi=75)
        fig.subplots_adjust(left=0,right=1,top=1,bottom=0,hspace=0,wspace=0)
        axes = axes.flatten()

        for i,(ax,objid) in enumerate(zip(axes,objlist)):

            if "seg" not in filt:
                img = fitsio.getdata("S82/fits/objID{:020d}-{:s}.fits".format(objid,filt))
                vmin,vmax = getVminVmax(img)
                ax.imshow(img,origin="lower",vmin=vmin,vmax=vmax,cmap=plt.cm.Greys)
            elif filt=="seg":
                img = fitsio.getdata("photom/galaxy/objID{:020d}-seg.fits".format(objid))
                img = reprocessSegMap(img)
                ax.imshow(img,origin="lower",cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(img))
            elif filt=="smthseg":
                img = fitsio.getdata("photom/smooth/objID{:020d}-seg.smooth.fits".format(objid))
                img = reprocessSegMap(img)
                ax.imshow(img,origin="lower",cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(img))

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        for ax in axes[i+1:]:
            ax.set_visible(False)

        fig.savefig("plots/summary_galaxy/stamps_galaxy_{:s}.png".format(filt))
        plt.close(fig)

    print("done.")

def chkGalaxyPhotom():

    catalog = fitsio.getdata('samples/final_clumpy_sample.fits')
    gal_cat = fitsio.getdata("photom/galaxy/galaxy_photom.fits")

    fig,axes = plt.subplots(3,2,figsize=(12,12),dpi=75)
    fig.subplots_adjust(left=0.06,right=0.98,bottom=0.05,top=0.98,wspace=0.15,hspace=0.025)
    axes[-1,-1].set_visible(False)
    axes = axes.T.flatten()[:-1]
    colors = ["purple","tab:blue","tab:green","orange","tab:red"]

    for ax,filt,color in zip(axes,sdss_filters,colors):

        ax.scatter(catalog["PETROMAG_%s"%filt],gal_cat["mag_auto_%s"%filt],marker='o',color=color,s=30,lw=0,alpha=0.8)
        ax.scatter(catalog["PETROMAG_%s"%filt],gal_cat["mag_iso_%s"%filt],marker='o',facecolor="none",edgecolor=color,s=25,lw=1.5,alpha=0.8)
        ax.text(0.02,0.98,"%s-band"%filt,color=color,fontsize=20,fontweight=600,ha="left",va="top",transform=ax.transAxes)
        ax.plot([-99,99],[-99,99],c='k')

        ax.set_xlim(21.9,13.1)
        ax.set_ylim(21.9,13.1)

        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        ax.set_ylabel("SDSS-DR7 mag [AB]",fontsize=16)
        [tick.set_fontsize(16) for tick in ax.get_xticklabels()+ax.get_yticklabels()]

    [tick.set_visible(False) for j in [0,1,3] for tick in axes[j].get_xticklabels()]

    axes[2].set_xlabel("SExtractor mag [AB]",fontsize=16)
    axes[4].set_xlabel("SExtractor mag [AB]",fontsize=16)
    fig.savefig("plots/sample/compare_galaxyPhotom.png".format(filt))

def main():

    catalog = fitsio.getdata('samples/final_clumpy_sample.fits')
    for i,entry in enumerate(catalog):
        print("\rProcessing {:d}/{:d} ... ".format(i+1,len(catalog)),end="")
        runSwarp(entry=entry,verbose=False)
        runSextractor(entry=entry,verbose=False)
    print("done.")

if __name__ == '__main__':

    # main()

    # mkGalaxyPhotom()
    chkSegmaps()
    chkGalaxyPhotom()

    # plt.show()

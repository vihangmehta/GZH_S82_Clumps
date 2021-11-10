from useful import *
from utils_S82 import getS82TsField

def mkVACatalog(overwriteTs=False):

    sample = fitsio.getdata("samples/gzh_sdss_clumpy.fits")
    sample = rfn.append_fields(sample,
                               names=["PSF_WIDTH_u","PSF_WIDTH_g","PSF_WIDTH_r","PSF_WIDTH_i","PSF_WIDTH_z","PSF_WIDTH_AVG","REDSHIFT_DR14"],
                               data=np.zeros((7,len(sample)))*np.NaN,
                               dtypes=float,usemask=False,asrecarray=False)

    for i,entry in enumerate(sample):

        ### Get DR14 redshift
        fname = "DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"])
        nearbyObjects = fitsio.getdata(fname)
        cond = np.isfinite(nearbyObjects["redshift"])
        if np.sum(cond)>0:
            sample[i]["REDSHIFT_DR14"] = nearbyObjects["redshift"][cond][0]

        ### Get PSF widths from the tsField files
        tsFieldname = getS82TsField(entry=entry,overwrite=overwriteTs)
        tsField = fitsio.getdata(tsFieldname)
        for j,filt in enumerate(sdss_filters):
            sample[i]["PSF_WIDTH_{:1s}".format(filt)] = tsField["PSF_WIDTH"][0,j]
        sample[i]["PSF_WIDTH_AVG"] = np.mean(tsField["PSF_WIDTH"][0,:])

    fitsio.writeto("samples/gzh_sdss_clumpy_VA.fits",sample,overwrite=True)

def curateFinalSample(overwrite=False,plot=False):

    ### Make the redshift cut
    sample = fitsio.getdata("samples/gzh_sdss_clumpy_VA.fits")
    sample = sample[np.isfinite(sample["REDSHIFT_DR14"])]
    sample = sample[(0<sample["REDSHIFT_DR14"]) & (sample["REDSHIFT_DR14"]<=0.06)]
    print("Un-curated clumpy sample: {:d}".format(len(sample)))
    if overwrite: np.savetxt("samples/curate_clumpy_sample.txt",sample["survey_id"],fmt="%20d")

    if plot:
        ### Make a summary file for checking
        filelist = " ".join(["plots/summary_sdss/objID{0[survey_id]:020d}.png".format(entry) for entry in sample])
        savename = "plots/pdf/curate_clumpy_sample.pdf"
        os.system("convert {0:s} {1:s}".format(filelist,savename))

def mkFinalSample(plot=False):

    sample = fitsio.getdata("samples/gzh_sdss_clumpy_VA.fits")
    finalIDs = np.genfromtxt("samples/curate_clumpy_sample.txt",dtype=int)

    sample = sample[np.in1d(sample["survey_id"], finalIDs)]
    print("Final cleaned clumpy sample: {:d}".format(len(sample)))
    fitsio.writeto("samples/final_clumpy_sample.fits", sample, overwrite=True)

    if plot:
        ### Make a summary PDF
        filelist = " ".join(["plots/summary_sdss/objID{0[survey_id]:020d}.png".format(entry) for entry in sample])
        savename = "plots/pdf/final_clumpy_sample.pdf"
        os.system("convert {0:s} {1:s}".format(filelist,savename))

if __name__ == '__main__':

    # mkVACatalog(overwriteTs=False)
    # curateFinalSample(overwrite=False,plot=False)
    mkFinalSample(plot=True)

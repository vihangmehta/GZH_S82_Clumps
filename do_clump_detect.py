from useful import *
from plotter import plotClumpsDetection

from astropy.convolution import Gaussian2DKernel,Box2DKernel,Tophat2DKernel,convolve,convolve_fft

def detectClumps(entry,smth=10,kernel_type="gauss",verbose=False):

    if kernel_type == "boxcar":
        kernel = Box2DKernel(smth)
        destdir = "boxcar{:d}".format(smth)
    elif kernel_type == "gauss":
        kernel = Gaussian2DKernel(x_stddev=smth, y_stddev=smth)
        destdir = "gauss{:d}".format(smth)
    elif kernel_type == "tophat":
        kernel = Tophat2DKernel(smth)
        destdir = "tophat{:d}".format(smth)
    else:
        raise Exception("Invalid kernel_type arg.")

    imgs = {}
    for filt in sdss_filters:

        img = "S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt)
        smoothed = "photom/{2:s}/objID{0[survey_id]:020d}-{1:s}.smooth.fits".format(entry,filt,destdir)
        contrast = "photom/{2:s}/objID{0[survey_id]:020d}-{1:s}.cntrst.fits".format(entry,filt,destdir)
        filtered = "photom/{2:s}/objID{0[survey_id]:020d}-{1:s}.filter.fits".format(entry,filt,destdir)

        img, hdr = fitsio.getdata(img, header=True)

        ################################################################
        ##### FIX for crazy bright stars right next to main object #####
        ################################################################
        size = img.shape[0]
        __img = img[int(np.floor(size / 4.0)) : int(np.ceil(size * 3.0 / 4.0)),
                    int(np.floor(size / 4.0)) : int(np.ceil(size * 3.0 / 4.0))]

        ### Just clip the outliers
        img = np.clip(img,np.median(img) - 10 * np.std(__img),
                          np.median(img) + 10 * np.std(__img))

        ### Replace the outliers with median value
        ### Not ideal because this creates holes in the centers of objects
        ### and causes unneeded deblending of sources
        # img[(np.median(img)-10*np.std(__img)>img) | (img>np.median(img)+10*np.std(__img))] = np.median(img)
        ################################################################

        med, sig = np.median(img), np.std(img)
        _img = img[(med - 3 * sig < img) & (img < med + 3 * sig)]
        bckgrnd = np.median(_img)

        # simg = convolve(img, kernel, fill_value=bckgrnd)
        simg = convolve_fft(img, kernel, fill_value=bckgrnd)
        cimg = img - simg

        fimg = cimg.copy()
        med, std = np.median(fimg), np.std(fimg)
        _fimg = fimg[(med - 3 * sig < fimg) & (fimg < med + 3 * sig)]
        fimg[(fimg < (np.median(_fimg) + 2 * np.std(_fimg)))] = 0

        fitsio.writeto(smoothed, data=simg, header=hdr, overwrite=True)
        fitsio.writeto(contrast, data=cimg, header=hdr, overwrite=True)
        fitsio.writeto(filtered, data=fimg, header=hdr, overwrite=True)

        imgs[filt] = filtered

    args = {"input_img": ",".join([imgs[_] for _ in detimg_filters]),
            "det_img": "photom/{1:s}/objID{0[survey_id]:020d}-det.fits".format(entry, destdir),
            "det_wht": "photom/{1:s}/objID{0[survey_id]:020d}-det.wht.fits".format(entry, destdir),
            "pixscale": FITS_pixscale,
            "img_size": ",".join(np.array(img.shape).astype("U5"))}

    call = "swarp {input_img:s} -c config/config_clump_detect.swarp " \
           "-WEIGHT_TYPE NONE " \
           "-IMAGEOUT_NAME {det_img:s} -WEIGHTOUT_NAME {det_wht:s} " \
           "-COMBINE Y -COMBINE_TYPE SUM -RESAMPLE N -SUBTRACT_BACK N " \
           "-PIXELSCALE_TYPE MANUAL -PIXEL_SCALE {pixscale:.4f} -IMAGE_SIZE {img_size:s}".format(**args)

    runBashCommand(call, cwd=cwd, verbose=verbose)
    os.remove(args["det_wht"])

    args = {"det_img": "photom/{1:s}/objID{0[survey_id]:020d}-det.fits".format(entry,destdir),
            "catname": "photom/{1:s}/objID{0[survey_id]:020d}-cat.fits".format(entry,destdir),
            "seg_img": "photom/{1:s}/objID{0[survey_id]:020d}-seg.fits".format(entry,destdir)}

    call = "sex {det_img:s} " \
           "-c config/config_clump_detect.sex " \
           "-PARAMETERS_NAME config/param_clump_detect.sex " \
           "-CATALOG_NAME {catname:s} -CATALOG_TYPE FITS_1.0 " \
           "-WEIGHT_TYPE NONE " \
           "-CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {seg_img:s}".format(**args)

    runBashCommand(call, cwd=cwd, verbose=verbose)

def main(sample):

    Parallel(n_jobs=10,verbose=10,backend="multiprocessing")(delayed(detectClumps)(entry,smth=10,kernel_type="gauss") for entry in sample)
    Parallel(n_jobs=10,verbose=10,backend="multiprocessing")(delayed(detectClumps)(entry,smth=15,kernel_type="gauss") for entry in sample)
    Parallel(n_jobs=10,verbose=10,backend="multiprocessing")(delayed(detectClumps)(entry,smth=10,kernel_type="boxcar") for entry in sample)
    Parallel(n_jobs=10,verbose=10,backend="multiprocessing")(delayed(detectClumps)(entry,smth=15,kernel_type="boxcar") for entry in sample)

    Parallel(n_jobs=15, verbose=10, backend="multiprocessing")(delayed(plotClumpsDetection)(entry,savefig=True) for entry in sample)

    filelist = " ".join(["plots/summary_clumps/objID{:020d}-clumps.png".format(i) for i in sample["survey_id"]])
    savename="plots/pdf/final_clumpy_detection.pdf"
    os.system("convert {0:s} {1:s}".format(filelist, savename))

def test(sample):

    entry = sample[sample["survey_id"] == 8647474690858025275][0]

    # detectClumps(entry, smth=10, kernel_type="gauss", verbose=False)
    # detectClumps(entry, smth=15, kernel_type="gauss", verbose=False)
    # detectClumps(entry, smth=10, kernel_type="boxcar", verbose=False)
    # detectClumps(entry, smth=15, kernel_type="boxcar", verbose=False)

    fig = plotClumpsDetection(entry=entry,savefig=False)
    # plt.savefig("plots/final/poster_detect.png")

def chkStats(sample):

    bins = 10 ** np.arange(-5, 5, 0.01)
    binc = 0.5 * (bins[1:] + bins[:-1])

    for destdir,c in zip(["boxcar10", "boxcar15", "gauss10", "gauss15"],
                         ["tab:blue", "tab:red", "tab:green", "orange"]):

        counts_det = np.zeros(0)
        counts_filt = np.zeros(0)

        for i, entry in enumerate(sample):

            img = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-det.fits".format(entry,destdir))
            counts_det = np.append(counts_det, img[img != 0])

            for filt in sdss_filters:
                img = fitsio.getdata("photom/{2:s}/objID{0[survey_id]:020d}-{1:s}.filter.fits".format(entry,filt,destdir))
                counts_filt = np.append(counts_filt, img[img != 0])

        hist = np.histogram(counts_det, bins=bins)[0]
        plt.plot(binc, hist, color=c, lw=1.5, ls="-", alpha=0.6, label=destdir)

        hist = np.histogram(counts_filt, bins=bins)[0]
        plt.plot(binc, hist, color=c, lw=1.5, ls="--", alpha=0.6)

    plt.legend()
    plt.xscale("log")

if __name__ == "__main__":

    sample = fitsio.getdata("samples/final_clumpy_sample.fits")

    # main(sample=sample)
    test(sample=sample)
    # chkStats(sample=sample)

    plt.show()

from useful import *

### Galaxy objIDs for special cases
poster  = [8647474690883846156, 8647474690337538126]
bristar = [8647474690339438847, 8647474691951296616,
           8647474691955294278, 8647475120883827059,
           8647475121992761658, 8647475121445077068]
blended = [8647475121457397936, 8647475122530025500,
           8647475122530025502]

def getClumpApertureSize(entry,units="pixel"):

    apersize = {"psf_avg": entry["PSF_WIDTH_AVG"],
                   "phot": entry["PSF_WIDTH_AVG"] * 1.5,
                   "ann0": entry["PSF_WIDTH_AVG"] * 2.0,
                   "ann1": entry["PSF_WIDTH_AVG"] * 3.0,
                   "mask": entry["PSF_WIDTH_AVG"] * 2.0,
                   "star": entry["PSF_WIDTH_AVG"] * 6.0}

    if units=="arcsec":
        return apersize
    elif units=="pixel":
        for x in apersize: apersize[x] /= FITS_pixscale
        return apersize
    else:
        raise Exception("Invalid units for getClumpApertureSize -- choose between 'pixel' and 'arcsec'")

def getClumpPositions(catalog):

    return np.vstack([catalog["X"]-1, catalog["Y"]-1]).T

def getClumpApertures(catalog, entry):

    apersize = getClumpApertureSize(entry=entry)

    pos  = getClumpPositions(catalog)
    aper = CircularAperture(pos,  r=0.5*apersize["phot"])
    annl = CircularAnnulus(pos,r_in=0.5*apersize["ann0"],
                              r_out=0.5*apersize["ann1"])

    return {"aper":aper,"annl":annl}

def getClumpMask(catalog,imshape,radius):

    pos  = getClumpPositions(catalog)
    mask = np.zeros(imshape, dtype=bool)

    for (xc, yc) in pos:
        c, r = np.indices(mask.shape)
        cond = ((r - xc)**2 + (c - yc)**2 <= radius**2)
        mask[cond] = 1
    return mask

def getStarMask(entry,imshape,radius):

    starList = np.genfromtxt("samples/custom_star_rejects.txt",dtype=[("objid",int),("X",float),("Y",float)])
    starList = starList[starList["objid"]==entry["survey_id"]]

    pos = np.vstack([starList["X"]-1, starList["Y"]-1]).T
    mask = np.zeros(imshape, dtype=bool)

    for (xc, yc) in pos:
        c, r = np.indices(mask.shape)
        cond = ((r - xc)**2 + (c - yc)**2 <= radius**2)
        mask[cond] = 1

    return mask

def calcReff(xc,yc,img,clumpMask,debug=False):

    minr, maxr = 0.5 * FITS_pixscale, np.sqrt(2) * max(img.shape) * FITS_pixscale
    radii = np.logspace(np.log10(minr), np.log10(maxr), int(np.ceil(maxr)))
    flux = np.zeros(len(radii))

    for i, (ri, ro) in enumerate(zip(radii[:-1], radii[1:])):
        aper = CircularAnnulus((xc - 1, yc - 1), r_in=ri, r_out=ro)
        mask = aper.to_mask(method="subpixel", subpixels=5)
        aimg = mask.multiply(img)
        aimg = aimg[(aimg >= 0) & (mask.data != 0)]
        if len(aimg) > 1:
            ### Only fill in the clumps, leave the seg edges alone
            new_area = len(aimg) + np.sum(mask.multiply(clumpMask))
            flux[i + 1] = flux[i] + np.mean(aimg) * new_area
        else:
            flux[i + 1] = flux[i]

    rcen = 0.5 * (radii[1:] + radii[:-1])
    flux = flux[1:] / np.max(flux)

    fint = scipy.interpolate.interp1d(rcen, flux, kind="cubic")
    reff = scipy.optimize.brentq(lambda x: fint(x) - 0.5, min(rcen), max(rcen) - 0.5)

    if debug:

        img = np.ma.masked_array(img, mask=img == -99)
        apers = [CircularAperture((xc - 1, yc - 1), r=r) for r in radii]
        _flux = aperture_photometry(img.filled(0), apers, method="subpixel", subpixels=5)
        _flux = [_flux["aperture_sum_{:d}".format(i)] for i in range(len(radii))]
        _flux = _flux / np.max(_flux)

        print("Reff:{:.2f}".format(reff))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=75, tight_layout=True)
        ax1.imshow(img, origin="lower")
        ax1.add_patch(Ellipse(xy=(xc-1,yc-1), width=2*reff, height=2*reff, angle=theta, edgecolor='c', facecolor='none', lw=1))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.plot(radii,_flux,c="k",marker="o",markersize=3,mew=0,label="w/o clump fill")
        ax2.plot(rcen,flux,c="tab:red",marker="o",markersize=3,mew=0,label="w/ clump fill")
        ax2.axhline(0.5, c="k", ls="--")
        ax2.axvline(reff, c="k", ls="--")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("Radius", fontsize=18)
        ax2.set_ylabel("Flux (<r) [norm]", fontsize=18)
        ax2.legend(fontsize=18)
        [_.set_fontsize(16) for _ in ax2.get_xticklabels() + ax2.get_yticklabels()]
        plt.show(block=True)

    return reff

def calcEllipseParameters(xc,yc,img,debug=False):

    y,x = np.indices(img.shape) + 1

    ### Second Moments
    x2 = np.ma.sum(img * x * x) / np.ma.sum(img) - xc**2
    y2 = np.ma.sum(img * y * y) / np.ma.sum(img) - yc**2
    xy = np.ma.sum(img * x * y) / np.ma.sum(img) - xc*yc

    sma = np.sqrt(0.5*(x2 + y2) + np.sqrt((0.5*(x2 - y2))**2 + xy**2))
    smb = np.sqrt(0.5*(x2 + y2) - np.sqrt((0.5*(x2 - y2))**2 + xy**2))
    theta = np.degrees(np.arctan(2*xy/(x2-y2)) / 2)

    if debug:

        print("SMA:{:.2f}, SMB:{:.2f}, THETA:{:.2f}".format(sma,smb,theta))
        fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=75,tight_layout=True)
        ax.imshow(img,origin="lower")
        ax.add_patch(Ellipse(xy=(xc-1,yc-1), width=2*sma, height=2*smb, angle=theta, edgecolor='r', facecolor='none', lw=1))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show(block=True)

    return sma,smb,theta

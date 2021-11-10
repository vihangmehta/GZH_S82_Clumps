from useful import *

import sfd
from extinction import calzetti00, ccm89
from astropy.coordinates import SkyCoord
from lmfit import Minimizer, Parameters, Parameter, report_fit

emLines = [(  "Ha_6563", 6562.80),
           (  "Hb_4861", 4861.33),
           (  "Hg_4340", 4340.47),
           ("OIII_5007", 5006.80),
           ("OIII_4959", 4958.92),
           ("OIII_4363", 4363.21),
           ( "OII_3727", 3727.00)]
emLines = OrderedDict(emLines)

def residuals(pars,wave,flux,ferr,center):

    model = emLineModel(wave,flux=pars["flux"],x0=center+pars["dlambda"],sig=pars["sigma"],cont=pars["cont"])
    resi  = (model - flux) / ferr
    return resi

def curveFitFunc(wave,dlambda,flux,sigma,cont,center):

    return emLineModel(wave,flux=flux,x0=center+dlambda,sig=sigma,cont=cont)

def lmfitFitter(wave,flux,ferr,redshift,center,flux_guess,sig_guess,cont_guess,verbose=False):

    pars = Parameters()
    pars["dlambda"]= Parameter(name="dlambda",value=0,          min=-10*(1+redshift), max=10*(1+redshift))
    pars["flux"]   = Parameter(name="flux",   value=flux_guess, min=0,                max=1e7)
    pars["sigma"]  = Parameter(name="sigma",  value=sig_guess,  min=0,                max=25)
    pars["cont"]   = Parameter(name="cont",   value=cont_guess)

    minner = Minimizer(residuals,pars,fcn_kws={"wave":wave,"flux":flux,"ferr":ferr,"center":center})
    result = minner.minimize()
    if verbose:
        report_fit(result)

    try:
        covar_diag = np.ma.masked_array(np.diag(result.covar),mask=np.diag(result.covar)<0,fill_value=np.NaN)
        result_errors = np.ma.sqrt(covar_diag)
    except AttributeError:
        result_errors = np.zeros_like(result.params,dtype=float) * np.NaN

    fit_par = OrderedDict([(_x,result.params[_x].value) for _x in result.params])
    fit_err = OrderedDict([(_x,_y) for _x,_y in zip(result.params,result_errors)])
    return fit_par, fit_err

def scipyFitter(wave,flux,ferr,redshift,center,flux_guess,sig_guess,cont_guess,verbose=False):

    try:
        func = lambda wave,dlambda,flux,sigma,cont: curveFitFunc(wave,dlambda,flux,sigma,cont,center=center)
        popt, pcov = scipy.optimize.curve_fit(func,xdata=wave,ydata=flux,sigma=ferr,absolute_sigma=True,
                                                   p0=[0,flux_guess,sig_guess,cont_guess],
                                                   bounds=([-10*(1+redshift),  0, 0,-1e5],
                                                           [ 10*(1+redshift),1e7,25, 1e5]))
        perr = np.sqrt(np.diag(pcov))
    except ValueError:
        popt, perr = np.zeros((2,4)) - np.nan

    fit_par = OrderedDict([(_x,_y) for _x,_y in zip(["dlambda","flux","sigma","cont"],popt)])
    fit_err = OrderedDict([(_x,_y) for _x,_y in zip(["dlambda","flux","sigma","cont"],perr)])
    return fit_par, fit_err

def fitEmLine(wave,flux,ferr,restw,redshift,
              window=100,flux_guess=1,sig_guess=3,
              fitter='lmfit',returnFull=False,verbose=False):

    center = restw * (1+redshift)

    emLineMask = np.zeros(len(wave),dtype=bool)
    for emline in emLines:
        emline_guess = emLines[emline]*(1+redshift)
        if np.abs(emline_guess - center) > 5:
            _cond = np.ma.abs(wave - emline_guess) < 10*(1+redshift)
            emLineMask[_cond] = 1

    wave,flux,ferr = wave.copy(),flux.copy(),ferr.copy()
    wave.mask = list((wave.mask | emLineMask).astype(int))
    flux.mask = list((flux.mask | emLineMask).astype(int))
    ferr.mask = list((ferr.mask | emLineMask).astype(int))

    cond = np.ma.abs(wave - center) < window/2.
    wave,flux,ferr = wave[cond],flux[cond],ferr[cond]
    cont_guess = np.ma.median(np.ma.clip(flux,np.ma.median(flux)-np.ma.std(flux),
                                              np.ma.median(flux)+np.ma.std(flux)))
    flux_guess = max(flux)/2 if len(flux)>10 else 1

    kwargs = {"wave":wave,"flux":flux,"ferr":ferr,"redshift":redshift,"center":center,
              "flux_guess":flux_guess,"sig_guess":sig_guess,"cont_guess":cont_guess,
              "verbose":verbose}

    if len(kwargs["wave"])<10 or sum(np.abs(kwargs["wave"]-center)<10*(1+redshift))<5:
        fit_par = OrderedDict([(_x,np.nan) for _x in ["dlambda","flux","sigma","cont"]])
        fit_err = OrderedDict([(_x,np.nan) for _x in ["dlambda","flux","sigma","cont"]])
    elif fitter=="lmfit":
        fit_par,fit_err = lmfitFitter(**kwargs)
    elif fitter=="scipy":
        fit_par,fit_err = scipyFitter(**kwargs)
    else:
        raise Exception("Invalid fitter argument")

    if returnFull:
        if any([np.isfinite(fit_par[x]) for x in fit_par]):
            _wave = np.linspace(min(wave),max(wave),len(wave)*100)
            _line = emLineModel(_wave,flux=fit_par["flux"],x0=center+fit_par["dlambda"],sig=fit_par["sigma"],cont=fit_par["cont"])
        else:
            _wave,_line = [],[]
        return fit_par,fit_err,_wave,_line

    return fit_par,fit_err

def getEmLines(catalog,fitter="scipy",debug=False):

    if not debug:
        fig, axes = {}, {}
        for emline in emLines:
            fig[emline],axes[emline] = plt.subplots(8,10,figsize=(10*1.5,8*1.5),dpi=100)
            fig[emline].subplots_adjust(left=0.005,right=0.995,bottom=0.01,top=0.99,wspace=0.05,hspace=0.05)
            axes[emline] = axes[emline].flatten()

    dtype = [("gzh_id",int),("clump_id",int),("clump_ra",float),("clump_dec",float),("clump_specobjid","U20")]
    for emline in emLines:
        dtype.extend([("{:s}_{:s}".format(emline,x),float) for x in ["SN","flux","fluxerr","flux_corr","fluxerr_corr","sig","sigerr","cont","conterr","dcen","dcenerr"]])

    emline_cat = np.recarray(len(catalog),dtype=dtype)
    for x in emline_cat.dtype.names: emline_cat[x] = -99
    emline_cat["gzh_id"]          = catalog["gzh_id"]
    emline_cat["clump_id"]        = catalog["clump_id"]
    emline_cat["clump_ra"]        = catalog["clump_ra"]
    emline_cat["clump_dec"]       = catalog["clump_dec"]
    emline_cat["clump_specobjid"] = catalog["clump_specobjid"]

    iax = 0
    for j,entry in enumerate(catalog):

        print("\rFitting {:d}/{:d} ... ".format(j+1,len(catalog)),end="",flush=True)
        if entry["clump_specobjid"]=='': continue

        redshift = entry["gal_redshiftdr14"]

        spec_data = fitsio.getdata("DR14/spectra/specID{:s}.fits".format(entry["clump_specobjid"].zfill(20)),1)
        mask = (spec_data["ivar"]==0)
        wave = np.ma.masked_array(10**spec_data["loglam"],mask=mask,fill_value=np.NaN)
        flux = np.ma.masked_array(spec_data["flux"],mask=mask,fill_value=np.NaN)
        ivar = np.ma.masked_array(spec_data["ivar"],mask=mask,fill_value=np.NaN)
        ferr = np.ma.sqrt(1/ivar)

        for emline in emLines:

            if emline=="OIII_4363":
                cond = np.ma.abs(wave - emLines["Hg_4340"]*(1+redshift)) < 10*(1+redshift)
                wave.mask = list((wave.mask | cond).astype(int))
                flux.mask = list((flux.mask | cond).astype(int))
                ferr.mask = list((ferr.mask | cond).astype(int))

            fit,err,_wave,_line = fitEmLine(wave=wave,flux=flux,ferr=ferr,
                                            restw=emLines[emline],redshift=redshift,
                                            window=150,fitter=fitter,
                                            returnFull=True,verbose=False)

            SN = fit["flux"]/err["flux"] if np.isfinite(err["flux"]) and err["flux"]>0 else 0
            emline_cat["{:s}_SN".format(emline)][j]      = SN
            emline_cat["{:s}_flux".format(emline)][j]    = fit["flux"]
            emline_cat["{:s}_fluxerr".format(emline)][j] = err["flux"]
            emline_cat["{:s}_sig".format(emline)][j]     = fit["sigma"]
            emline_cat["{:s}_sigerr".format(emline)][j]  = err["sigma"]
            emline_cat["{:s}_cont".format(emline)][j]    = fit["cont"]
            emline_cat["{:s}_conterr".format(emline)][j] = err["cont"]
            emline_cat["{:s}_dcen".format(emline)][j]    = fit["dlambda"]
            emline_cat["{:s}_dcenerr".format(emline)][j] = err["dlambda"]

            if debug:
                fig,ax = plt.subplots(1,1)
                ax.text(0.98,0.98,emline,color='k',va="top",ha="right",fontsize=16,fontweight=600,transform=ax.transAxes)
            else:
                ax = axes[emline][iax]

            ax.plot(wave,flux,c='k')
            ax.fill_between(wave,flux-ferr,flux+ferr,color='k',lw=0,alpha=0.2)
            ax.plot(_wave,_line,c='g' if SN>3 else 'r')
            ax.axvline(emLines[emline]*(1+redshift),c='k',ls='--',lw=0.8)
            ax.text(0.02,0.98,"{:.2f}".format(SN),color='g' if SN>3 else 'r',
                        va='top',ha='left',fontsize=16,fontweight=600,transform=ax.transAxes)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            ocen = emLines[emline]*(1+redshift)
            cond = np.ma.abs(wave - ocen) < 200/2.
            ax.set_xlim(ocen-100,ocen+100)
            ax.set_ylim(min(flux[cond])/1.2,max(flux[cond])*1.2)

            if debug:
                plt.show()

        iax += 1

    print("done.")

    for emline in emLines:
        for ax in axes[emline][iax:]:
            ax.set_visible(False)
        fig[emline].savefig("plots/spec/emline_{:s}_{:s}.png".format(fitter,emline))

    emline_cat = correctEmLinesForGalExtinction(emline_cat)
    fitsio.writeto("samples/final_clump_linecat_{:s}.fits".format(fitter),emline_cat,overwrite=True)

def getSDSSEmLines(catalog):

    kdict = {  "Ha_6563" : "H_alpha",
               "Hb_4861" : "H_beta",
               "Hg_4340" : "H_gamma",
             "OIII_5007" : "[O_III] 5007",
             "OIII_4959" : "[O_III] 4959",
             "OIII_4363" : "[O_III] 4363",
              "OII_3727" : "[O_II] 3727"}

    dtype = [("gzh_id",int),("clump_id",int),("clump_ra",float),("clump_dec",float),("clump_specobjid","U20")]
    for emline in emLines:
        dtype.extend([("{:s}_{:s}".format(emline,x),float) for x in ["SN","flux","fluxerr","flux_corr","fluxerr_corr","sig","sigerr","cont","conterr","z","zerr","ew","ewerr"]])

    emline_cat = np.recarray(len(catalog),dtype=dtype)
    for x in emline_cat.dtype.names: emline_cat[x] = -99
    emline_cat["gzh_id"]          = catalog["gzh_id"]
    emline_cat["clump_id"]        = catalog["clump_id"]
    emline_cat["clump_ra"]        = catalog["clump_ra"]
    emline_cat["clump_dec"]       = catalog["clump_dec"]
    emline_cat["clump_specobjid"] = catalog["clump_specobjid"]

    for j,entry in enumerate(catalog):

        if entry["clump_specobjid"]=='': continue

        spec_data = fitsio.open("DR14/spectra/specID{:s}.fits".format(entry["clump_specobjid"].zfill(20)))

        for emline in emLines:

            idx = np.where(spec_data[3].data["LINENAME"] == kdict[emline])[0][0]
            sdss_entry = spec_data[3].data[idx]

            emline_cat[j]["{:s}_SN".format(emline)]      = sdss_entry["LINEAREA"] / sdss_entry["LINEAREA_ERR"]
            emline_cat[j]["{:s}_flux".format(emline)]    = sdss_entry["LINEAREA"]
            emline_cat[j]["{:s}_fluxerr".format(emline)] = sdss_entry["LINEAREA_ERR"]
            emline_cat[j]["{:s}_sig".format(emline)]     = sdss_entry["LINESIGMA"]
            emline_cat[j]["{:s}_sigerr".format(emline)]  = sdss_entry["LINESIGMA_ERR"]
            emline_cat[j]["{:s}_cont".format(emline)]    = sdss_entry["LINECONTLEVEL"]
            emline_cat[j]["{:s}_conterr".format(emline)] = sdss_entry["LINECONTLEVEL_ERR"]
            emline_cat[j]["{:s}_z".format(emline)]       = sdss_entry["LINEZ"]
            emline_cat[j]["{:s}_zerr".format(emline)]    = sdss_entry["LINEZ_ERR"]
            emline_cat[j]["{:s}_ew".format(emline)]      = sdss_entry["LINEEW"]
            emline_cat[j]["{:s}_ewerr".format(emline)]   = sdss_entry["LINEEW_ERR"]

            if emline=="OII_3727":
                idx = np.where(spec_data[3].data["LINENAME"] == "[O_II] 3725")[0][0]
                sdss_entry = spec_data[3].data[idx]
                emline_cat[j]["{:s}_flux".format(emline)]   += sdss_entry["LINEAREA"]
                emline_cat[j]["{:s}_fluxerr".format(emline)] = np.sqrt(emline_cat[j]["{:s}_fluxerr".format(emline)]**2 + \
                                                                       sdss_entry["LINEAREA_ERR"]**2)

    emline_cat = correctEmLinesForGalExtinction(emline_cat)
    fitsio.writeto("samples/final_clump_linecat_sdss.fits",emline_cat,overwrite=True)

def correctEmLinesForGalExtinction(catalog):

    waves = np.array(list(emLines.values()))
    for entry in catalog:

        ### Correct for galactic dust
        ra,dec = np.atleast_1d(entry["clump_ra"]), np.atleast_1d(entry["clump_dec"])
        coords = SkyCoord(ra =np.atleast_1d(ra )*u.degree,
                          dec=np.atleast_1d(dec)*u.degree)
        EBV = sfd.ebv(coords)
        AV = 3.1 * EBV
        Almbda = ccm89(waves, a_v=AV, r_v=3.1)

        for emline,Al in zip(emLines.keys(),Almbda):
            entry["{:s}_flux_corr".format(emline)]    = entry["{:s}_flux".format(emline)] * 10**(0.4*Al)
            entry["{:s}_fluxerr_corr".format(emline)] = entry["{:s}_fluxerr".format(emline)] * 10**(0.4*Al)

        ### Correct for host galaxy dust
        Ha, Hb = entry["Ha_6563_flux_corr"], entry["Hb_4861_flux_corr"]
        EBV = 1.97 * np.log10(Ha/Hb/2.86) if Ha/Hb > 2.86 else 0.0
        AV = 4.05 * EBV
        Almbda = calzetti00(waves, a_v=AV, r_v=4.05)

        for emline,Al in zip(emLines.keys(),Almbda):
            entry["{:s}_flux_corr".format(emline)]    = entry["{:s}_flux".format(emline)] * 10**(0.4*Al)
            entry["{:s}_fluxerr_corr".format(emline)] = entry["{:s}_fluxerr".format(emline)] * 10**(0.4*Al)

    return catalog

def compareFluxes():

    fit1cat = fitsio.getdata("samples/final_clump_linecat_lmfit.fits")
    fit2cat = fitsio.getdata("samples/final_clump_linecat_scipy.fits")
    sdsscat = fitsio.getdata("samples/final_clump_linecat_sdss.fits")

    fig,axes = plt.subplots(2,3,figsize=(18,9),dpi=75,tight_layout=True)
    axes = axes.flatten()

    for ax,emline in zip(axes,emLines):

        cond = sdsscat["{:s}_flux".format(emline)] > 0

        ax.scatter( fit1cat["{:s}_flux".format(emline)][cond],
                   (fit1cat["{:s}_flux".format(emline)][cond] - sdsscat["{:s}_flux".format(emline)][cond])/fit1cat["{:s}_flux".format(emline)][cond],
                        color='tab:red',s=15,alpha=0.8,label="LMFIT")

        ax.scatter( fit2cat["{:s}_flux".format(emline)][cond],
                   (fit2cat["{:s}_flux".format(emline)][cond] - sdsscat["{:s}_flux".format(emline)][cond])/fit2cat["{:s}_flux".format(emline)][cond],
                        color='k',s=15,alpha=0.8,label="SciPy")

        # ax.plot([1e-5,1e10],[1e-5,1e10],c='k',ls='--',lw=0.8)
        # ax.set_ylim(1e-1,1e4)
        ax.axhline(0,ls='--',c='k',lw=0.5)
        ax.set_ylim(-2,2)
        ax.set_xlim(1e-1,1e4)
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_title(emline,fontsize=18)

    axes[0].set_ylabel("$\\Delta(f)/f$",fontsize=16)
    axes[3].set_ylabel("$\\Delta(f)/f$",fontsize=16)
    axes[-2].set_xlabel("flux",fontsize=16)
    axes[-1].legend(fontsize=18)
    plt.show()

def mkFinalSpecCat():

    emlncat = fitsio.getdata("samples/final_clump_linecat_scipy.fits")
    sdsscat = fitsio.getdata("samples/final_clump_linecat_sdss.fits")

    for x in sdsscat.dtype.names:
        if "OIII_4363_" in x:
            if x in emlncat.dtype.names:
                sdsscat[x] = emlncat[x]
            else:
                sdsscat[x] = -99

    fitsio.writeto("samples/final_clump_linecat.fits",sdsscat,overwrite=True)

if __name__ == '__main__':

    catalog = fitsio.getdata("samples/final_clump_catalog.fits")

    # getEmLines(catalog=catalog,fitter="lmfit",debug=False)
    # getEmLines(catalog=catalog,fitter="scipy",debug=False)
    # getSDSSEmLines(catalog=catalog)

    compareFluxes()
    # mkFinalSpecCat()

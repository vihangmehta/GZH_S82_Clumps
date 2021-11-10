import numpy as np
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import scipy.optimize as so
import pyneb as pn

O2_atom = pn.Atom("O",2)
O3_atom = pn.Atom("O",3)

def get_line_ratios(cat_emlines):

    o4959, o4959err, o4959_sn = cat_emlines["OIII_4959_flux_corr"] ,cat_emlines["OIII_4959_fluxerr_corr"] , cat_emlines["OIII_4959_SN"]
    o5007, o5007err, o5007_sn = cat_emlines["OIII_5007_flux_corr"] ,cat_emlines["OIII_5007_fluxerr_corr"] , cat_emlines["OIII_5007_SN"]
    o4363, o4363err, o4363_sn = cat_emlines["OIII_4363_flux_corr"] ,cat_emlines["OIII_4363_fluxerr_corr"] , cat_emlines["OIII_4363_SN"]
    o3727, o3727err, o3727_sn = cat_emlines["OII_3727_flux_corr"]  ,cat_emlines["OII_3727_fluxerr_corr"]  , cat_emlines["OII_3727_SN"]
    hbeta, hbetaerr, hbeta_sn = cat_emlines["Hb_4861_flux_corr"]   ,cat_emlines["Hb_4861_fluxerr_corr"]   , cat_emlines["Hb_4861_SN"]

    R23,O3,O32,R3,P = np.zeros((5,len(cat_emlines))) - 99.
    R23err,O3err,O32err,R3err,Perr = np.zeros((5,len(cat_emlines)))

    cond_R23 = (o3727_sn>3) & (o4959_sn>3) & (o5007_sn>3) & (hbeta_sn>3)
    cond_O3  = (o4959_sn>3) & (o5007_sn>3) & (o4363_sn>3)
    cond_O32 = (o4959_sn>3) & (o5007_sn>3) & (o3727_sn>3)
    cond_R3  = (o4959_sn>3) & (o5007_sn>3) & (hbeta_sn>3)
    cond_P   = cond_R3 & cond_R23

    R23[cond_R23] = np.log10((o3727+o4959+o5007)[cond_R23] / hbeta[cond_R23])
    O3[ cond_O3 ] = np.log10((o4959+o5007)[cond_O3] / o4363[cond_O3])
    O32[cond_O32] = np.log10((o4959+o5007)[cond_O32] / o3727[cond_O32])
    R3[ cond_R3 ] = np.log10((o4959+o5007)[cond_R3] / hbeta[cond_R3])
    P[  cond_P  ] = R3[cond_P] - R23[cond_P]

    R23err[  cond_R23]   = np.abs(R23[  cond_R23]  ) * np.sqrt((o3727err/o3727)[cond_R23  ]**2+(o4959err/o4959)[cond_R23  ]**2+(o5007err/o5007)[cond_R23]**2+(hbetaerr/hbeta)[cond_R23]**2)
    O3err[   cond_O3]    = np.abs(O3[   cond_O3]   ) * np.sqrt((o4363err/o4363)[cond_O3   ]**2+(o4959err/o4959)[cond_O3   ]**2+(o5007err/o5007)[cond_O3 ]**2)
    O32err[  cond_O32]   = np.abs(O32[  cond_O32]  ) * np.sqrt((o3727err/o3727)[cond_O32  ]**2+(o4959err/o4959)[cond_O32  ]**2+(o5007err/o5007)[cond_O32]**2)
    R3err[   cond_R3]    = np.abs(R3[   cond_R3]   ) * np.sqrt((o4959err/o4959)[cond_R3   ]**2+(o5007err/o5007)[cond_R3   ]**2+(hbetaerr/hbeta)[cond_R3 ]**2)

    return {"R23":R23,"O3":O3,"O32":O32,"R3":R3}, {"R23":R23err,"O3":O3err,"O32":O32err,"R3":R3err}

def mass_metallicity_relation_T04(logMst):

    logOH = -1.492 + 1.847*logMst - 0.08026*logMst**2
    return logOH

def mass_metallicity_relation_B12(logMst,return_scatter=False):

    if return_scatter:
        logOH_lower = (5.61-0.24) + (0.29+0.03) * logMst
        logOH_upper = (5.61+0.24) + (0.29-0.03) * logMst
        return logOH_lower,logOH_upper
    logOH = 5.61 + 0.29 * logMst
    return logOH

def direct_method(entry):
    """
    References:
        Izotov+06 -- https://arxiv.org/pdf/astro-ph/0511644.pdf
        Ly+13     -- https://arxiv.org/pdf/1307.7712.pdf
    """

    o4959 = entry["OIII_4959_flux_corr"]
    o5007 = entry["OIII_5007_flux_corr"]
    o4363 = entry["OIII_4363_flux_corr"]
    o3727 = entry["OII_3727_flux_corr"]
    hbeta = entry["Hb_4861_flux_corr"]

    if (entry["OIII_4959_SN"]>3) and \
       (entry["OIII_5007_SN"]>3) and \
       (entry["OIII_4363_SN"]>3) and \
       (entry[ "OII_3727_SN"]>3) and \
       (entry[  "Hb_4861_SN"]>3):

        O3 = np.log10((o4959+o5007) / o4363)
        R2 = np.log10( o3727 / hbeta)
        R3 = np.log10((o4959+o5007) / hbeta)

        CT   = lambda t: 8.44 - 1.09*t + 0.5*t**2 -0.08*t**3
        func = lambda t: 1.432 / (O3 - np.log10(CT(t))) - t

        try:
            t = so.brentq(func,1e-5,6.60531)
        except ValueError:
            print(o4959,o5007,o4363,o3727,hbeta)
            return np.NaN, np.NaN, np.NaN

        logTe = np.log10(t * 1e4)
        logOH_II  = R2 + 5.961 + 1.676/t - 0.40*np.log10(t) - 0.034*t
        logOH_III = R3 + 6.2   + 1.251/t - 0.55*np.log10(t) - 0.014*t

        OH = 10**(logOH_II-12) + 10**(logOH_III-12)
        logOH = 12 + np.log10(OH)

        Z_neb = 29 * 10**(logOH-12)     # Ref: Kobulnicky & Kewley (2004) -- https://arxiv.org/pdf/astro-ph/0408128.pdf

        return logTe, logOH, Z_neb

    else:

        return -99, -99, -99

def direct_method_pyneb(entry, O3_den=1e2, simulate_noise=False, seed=None):

    o4959 = entry["OIII_4959_flux_corr"]
    o5007 = entry["OIII_5007_flux_corr"]
    o4363 = entry["OIII_4363_flux_corr"]
    o3727 = entry["OII_3727_flux_corr"]
    hbeta = entry["Hb_4861_flux_corr"]

    if simulate_noise:
        if seed is not None: np.random.seed(seed)
        else: print("Warning invalid seed.")
        o4959 += np.random.normal(scale=entry["OIII_4959_fluxerr_corr"])
        o5007 += np.random.normal(scale=entry["OIII_5007_fluxerr_corr"])
        o4363 += np.random.normal(scale=entry["OIII_4363_fluxerr_corr"])
        o3727 += np.random.normal(scale=entry["OII_3727_fluxerr_corr"])
        hbeta += np.random.normal(scale=entry["Hb_4861_fluxerr_corr"])

    if((entry["OIII_4959_SN"]>10) and \
       (entry["OIII_5007_SN"]>10) and \
       (entry["OIII_4363_SN"]> 3) and \
       (entry[ "OII_3727_SN"]>10) and \
       (entry[  "Hb_4861_SN"]>10)) or simulate_noise:

        O3 = (o4959+o5007) / o4363
        R2 = (o3727 / hbeta)
        R3 = (o4959+o5007) / hbeta

        O3_tem = O3_atom.getTemDen(O3, den=O3_den, to_eval="(L(4959)+L(5007))/L(4363)")
        O3_abund = O3_atom.getIonAbundance(R3, tem=O3_tem, den=O3_den, to_eval="L(4959)+L(5007)", Hbeta=1)
        O2_abund = O2_atom.getIonAbundance(R2, tem=O3_tem, den=O3_den, to_eval="L(3726)+L(3729)", Hbeta=1)

        logTe = np.log10(O3_tem)
        logOH = 12 + np.log10(O3_abund + O2_abund)
        Z_neb = 29 * 10**(logOH-12)

        return logTe, logOH, Z_neb

    else:

        return -99, -99, -99

def indirect_method(entry,calib="P05",simulate_noise=False,seed=None):

    o4959 = entry["OIII_4959_flux_corr"]
    o5007 = entry["OIII_5007_flux_corr"]
    o3727 = entry["OII_3727_flux_corr"]
    hbeta = entry["Hb_4861_flux_corr"]

    if simulate_noise:
        if seed is not None: np.random.seed(seed)
        else: print("Warning invalid seed.")
        o4959 += np.random.normal(scale=entry["OIII_4959_fluxerr_corr"])
        o5007 += np.random.normal(scale=entry["OIII_5007_fluxerr_corr"])
        o3727 += np.random.normal(scale=entry["OII_3727_fluxerr_corr"])
        hbeta += np.random.normal(scale=entry["Hb_4861_fluxerr_corr"])

    if((entry["OIII_4959_SN"]>10) and \
       (entry["OIII_5007_SN"]>10) and \
       (entry[ "OII_3727_SN"]>10) and \
       (entry[  "Hb_4861_SN"]>10)) or simulate_noise:

        R23 = np.log10((o3727+o4959+o5007) / hbeta)
        O32 = np.log10((o4959+o5007) / o3727)
        R3  = np.log10((o4959+o5007) / hbeta)
        P   = R3 - R23

        if calib=="P05":
            # Reference: Pilyugin & Thuan (2005) -- https://iopscience.iop.org/article/10.1086/432408/pdf
            # Lower branch calibration for 12+log(O/H) < 8
            func_P05_upper = lambda R23,P: (R23 + 726.1 + 842.2*P + 337.5*P**2) / (85.96 + 82.76*P + 43.98*P**2 + 1.793*R23)
            func_P05_lower = lambda R23,P: (R23 + 106.4 + 106.8*P -   3.4*P**2) / (17.72 +  6.6 *P +  6.95*P**2 - 0.302*R23)
            logOH_upper = func_P05_upper(10**R23,10**P)
            logOH_lower = func_P05_lower(10**R23,10**P)
            # print("P05",logOH_upper,logOH_lower)
            return logOH_upper, logOH_lower

        elif calib=="J15":
            # Reference: Jones+15 -- https://arxiv.org/pdf/1504.02417.pdf
            func_J15 = lambda logOH: -54.1003 + 13.9083*logOH - 0.8782*logOH**2
            xx = np.arange(7.5,9,1e-4)
            max_R23 = max(func_J15(xx))
            if (R23 >= max_R23) and (R23 - max_R23 < 0.06):
                logOH = xx[np.argmax(func_J15(xx))]
                # print("J15",logOH)
                return logOH, logOH
            elif R23 < max_R23:
                logOH_upper = so.brentq(lambda x:func_J15(x) - R23,5,7.919)
                logOH_lower = so.brentq(lambda x:func_J15(x) - R23,7.919,12)
                # print("J15",logOH_upper,logOH_lower)
                return logOH_upper, logOH_lower
            else:
                # print("J15",-99)
                return -99, -99

        else:
            print ("Invalid calib.")

    else:
        return -99, -99

def comparePyNeb(linecat):

    logTe,logOH_dir,Z_dir = np.zeros((3,len(linecat)),dtype=float) - 99
    logTe_pn,logOH_dir_pn,Z_dir_pn = np.zeros((3,len(linecat)),dtype=float) - 99

    for i,entry in enumerate(linecat):
        logTe[i], logOH_dir[i], Z_dir[i] = direct_method(entry)
        logTe_pn[i], logOH_dir_pn[i], Z_dir_pn[i] = direct_method_pyneb(entry)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,7),dpi=75,tight_layout=True)

    ax1.scatter(logOH_dir,logOH_dir_pn,c='k',s=15,alpha=0.8)
    ax1.plot([-99,99],[-99,99],c='k',lw=0.8,alpha=0.8)
    ax1.set_xlabel("log OH (manual)",fontsize=20)
    ax1.set_ylabel("log OH (PyNeb)",fontsize=20)
    ax1.set_xlim(7.2,8.4)
    ax1.set_ylim(7.2,8.4)
    ax1.set_aspect(1)

    ax2.scatter(logTe,logTe_pn,c='k',s=15,alpha=0.8)
    ax2.plot([-99,99],[-99,99],c='k',lw=0.8,alpha=0.8)
    ax2.set_xlabel("log T$_e$ (manual)",fontsize=20)
    ax2.set_ylabel("log T$_e$ (PyNeb)",fontsize=20)
    ax2.set_xlim(3.9,4.4)
    ax2.set_ylim(3.9,4.4)
    ax2.set_aspect(1)

    [_.set_fontsize(14) for _ in ax1.get_xticklabels()+ax1.get_yticklabels()+ax2.get_xticklabels()+ax2.get_yticklabels()]

    cond = (logOH_dir_pn<7.5) & (logOH_dir_pn>0)
    print(linecat["gzh_id"][cond],linecat["clump_id"][cond],logOH_dir_pn[cond])

if __name__ == '__main__':

    linecat = fitsio.getdata("samples/final_clump_linecat.fits")

    # for entry in linecat:
    #     print(entry["ID"],direct_method(entry))
    #     print(entry["ID"],indirect_method(entry))

    comparePyNeb(linecat)
    plt.show()

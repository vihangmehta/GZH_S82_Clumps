from useful import *

def transformImgToWCS(ra, dec, ra0, dec0, size, scale):

    x = (-(ra - ra0) * 3600.0 + size / 2.0) / scale
    y = (-(dec - dec0) * 3600.0 + size / 2.0) / scale
    return x, y

def getAngularSize(z=np.NaN, physSize=None, angSize=None):

    if physSize is not None and np.isfinite(z):
        if z > 0:
            ang_diameter_dist = (Planck15.angular_diameter_distance(z).value * 1e3)  # in kpc
            angSize = physSize / ang_diameter_dist
            angSize = angSize / np.pi * 180 * 3600.0
        else:
            return np.NaN
    elif angSize is not None:
        angSize = angSize
    else:
        raise Exception("Invalid args for getAngularSize().")
    return angSize

def getCutoutSize(entry):

    nearbyObjectsFile = "DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"])

    use_z = entry["REDSHIFT"]
    if not np.isfinite(use_z) and os.path.isfile(nearbyObjectsFile):
        nearbyObjects = fitsio.getdata(nearbyObjectsFile)
        nearbyObject_z = nearbyObjects["redshift"][0]
        if np.isfinite(nearbyObject_z):
            use_z = nearbyObject_z

    cutout_size = max(5*getAngularSize(z=use_z,physSize=15,angSize=10), 30)
    cutout_size = min(cutout_size, 120)
    return cutout_size

def querySpecSearch(ra,dec,radius):

    specSearch = fitsio.getdata("DR14/specSearch/gzh_sdss_clumpy_specSearch.fits")

    m1,m2,dist = matchRADec(ra,dec,specSearch["ra"],specSearch["dec"],crit=radius,maxmatch=0)
    nearbyObjects = specSearch[m2]
    isort = np.argsort(dist)
    nearbyObjects, dist = nearbyObjects[isort], dist[isort]

    return nearbyObjects, dist

def extractSwarpCutout(img,ra,dec,size,savename,overwrite=False,verbose=False):

    def __run_swarp(args):
        while os.path.isfile(os.path.join(cwd,".tmp",args["input_img"].split("/")[-1].replace(".fits",".resamp.fits"))) or \
              os.path.isfile(os.path.join(cwd,".tmp",args["input_img"].split("/")[-1].replace(".fits",".resamp.weight.fits"))):
            time.sleep(2)
        runBashCommand(call,cwd=cwd,verbose=verbose)
        os.remove(args["out_wht"])

    size = int(np.ceil(size / FITS_pixscale))
    args = {"input_img": img,
            "out_img": savename,
            "out_wht": savename.replace(".fits",".wht.fits"),
            "pixscale": FITS_pixscale,
            "center": "{:.8f},{:.8f}".format(ra,dec),
            "img_size": "{:d},{:d}".format(size,size)}

    call = "swarp {input_img:s} -c config/config_stamp.swarp " \
           "-IMAGEOUT_NAME {out_img:s} -WEIGHTOUT_NAME {out_wht:s} " \
           "-RESCALE_WEIGHTS N -COMBINE Y -RESAMPLE Y -SUBTRACT_BACK N " \
           "-CENTER_TYPE MANUAL -CENTER {center:s} " \
           "-PIXELSCALE_TYPE MANUAL -PIXEL_SCALE {pixscale:.4f} -IMAGE_SIZE {img_size:s}".format(**args)

    if overwrite: __run_swarp(args)
    file_readable = False
    while not file_readable:
        try:
            _ = fitsio.getdata(args["out_img"])
            file_readable = True
        except (OSError,TypeError):
            __run_swarp(args)

def mkSummaryPhotObj(catalog):

    nearbyPhotObjectsAll = fitsio.getdata("DR14/photObj/nearby_photObj_ID{:020d}.fits".format(catalog[0]["survey_id"]))

    for entry in catalog[1:]:
        nearbyPhotObjects = fitsio.getdata("DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"]))
        nearbyPhotObjectsAll = rfn.stack_arrays((nearbyPhotObjectsAll,nearbyPhotObjects),usemask=False,asrecarray=True)

    iunique = np.unique(nearbyPhotObjectsAll["objid"],return_index=True)[1]
    nearbyPhotObjectsAll = nearbyPhotObjectsAll[iunique]
    fitsio.writeto("DR14/photObj/nearby_photObjAll.fits",nearbyPhotObjectsAll,overwrite=True)

def mkSummarySpecObj(catalog):

    cond = [os.path.isfile("DR14/specObj/nearby_specObj_ID{:020d}.fits".format(entry["survey_id"])) for entry in catalog]
    catalog = catalog[cond]

    nearbySpecObjectsAll = fitsio.getdata("DR14/specObj/nearby_specObj_ID{:020d}.fits".format(catalog[0]["survey_id"]))

    for entry in catalog[1:]:
        nearbySpecObjects = fitsio.getdata("DR14/specObj/nearby_specObj_ID{:020d}.fits".format(entry["survey_id"]))
        nearbySpecObjectsAll = rfn.stack_arrays((nearbySpecObjectsAll,nearbySpecObjects),usemask=False,asrecarray=True)

    iunique = np.unique(nearbySpecObjectsAll["specobjid"],return_index=True)[1]
    nearbySpecObjectsAll = nearbySpecObjectsAll[iunique]
    fitsio.writeto("DR14/specObj/nearby_specObjAll.fits",nearbySpecObjectsAll,overwrite=True)

def getTotalNClumps(sample,destdir="boxcar10",suffix="phot"):

    N = 0
    for entry in sample:
        try:
            N+=len(fitsio.getdata(os.path.join(cwd,"photom/{1:s}/objID{0[survey_id]:020d}-{2:s}.fits".format(entry, destdir, suffix))))
        except OSError:
            pass
    return N

def clumpCatalogDtype():

    gzh_cat_dtype  = [('gzh_id',              'survey_id',       '>i8'),
                      ('gzh_clumpy_selection','clumpy_selection', 'U6'),
                      ('gzh_id_single',       'survey_id_single','>i8'),
                      ('gzh_ra',              'RA',              '>f8'),
                      ('gzh_dec',             'DEC',             '>f8'),
                      ('gzh_run',             'RUN',             '>i2'),
                      ('gzh_rerun',           'RERUN',           '>i2'),
                      ('gzh_camcol',          'CAMCOL',          '>i2'),
                      ('gzh_field',           'FIELD',           '>i2'),
                      ('gzh_obj',             'OBJ',             '>i2'),
                      ('gzh_petror50_r',      'PETROR50_R',      '>f4'),
                      ('gzh_petror90_r',      'PETROR90_R',      '>f4'),
                      ('gzh_mag_u',           'PETROMAG_U',      '>f4'),
                      ('gzh_magerr_u',        'PETROMAGERR_U',   '>f4'),
                      ('gzh_mag_g',           'PETROMAG_G',      '>f4'),
                      ('gzh_magerr_g',        'PETROMAGERR_G',   '>f4'),
                      ('gzh_mag_r',           'PETROMAG_R',      '>f4'),
                      ('gzh_magerr_r',        'PETROMAGERR_R',   '>f4'),
                      ('gzh_mag_i',           'PETROMAG_I',      '>f4'),
                      ('gzh_magerr_i',        'PETROMAGERR_I',   '>f4'),
                      ('gzh_mag_z',           'PETROMAG_Z',      '>f4'),
                      ('gzh_magerr_z',        'PETROMAGERR_Z',   '>f4'),
                      ('gzh_extinction_r',    'EXTINCTION_R',    '>f4'),
                      ('gzh_psf_width_u',     'PSF_WIDTH_u',     '>f4'),
                      ('gzh_psf_width_g',     'PSF_WIDTH_g',     '>f4'),
                      ('gzh_psf_width_r',     'PSF_WIDTH_r',     '>f4'),
                      ('gzh_psf_width_i',     'PSF_WIDTH_i',     '>f4'),
                      ('gzh_psf_width_z',     'PSF_WIDTH_z',     '>f4'),
                      ('gzh_redshift',        'REDSHIFT',        '>f4'),
                      ('gzh_redshifterr',     'REDSHIFTERR',     '>f4'),
                      ('gal_redshiftdr14',    'REDSHIFT_DR14',   '>f4'),
                      ('gal_specobjid',        None,             'U20')]

    gal_imag_dtype = [('gal_x',               'GAL_XC',          '>f4'),
                      ('gal_y',               'GAL_YC',          '>f4'),
                      ('gal_ra',              'GAL_RA',          '>f8'),
                      ('gal_dec',             'GAL_DEC',         '>f8'),
                      ('gal_reff',            'GAL_REFF',        '>f4'),
                      ('gal_reff_xy',         'GAL_REFF_XY',     '>f4'),
                      ('gal_sma',             'GAL_SMA',         '>f4'),
                      ('gal_smb',             'GAL_SMB',         '>f4'),
                      ('gal_theta',           'GAL_THETA',       '>f4'),
                      ('gal_sma_xy',          'GAL_SMA_XY',      '>f4'),
                      ('gal_smb_xy',          'GAL_SMB_XY',      '>f4')]

    gal_phot_dtype = [('gal_flux_u',          'flux_auto_u',     '>f4'),
                      ('gal_fluxerr_u',       'fluxerr_auto_u',  '>f4'),
                      ('gal_mag_u',           'mag_auto_u',      '>f4'),
                      ('gal_magerr_u',        'magerr_auto_u',   '>f4'),
                      ('gal_flux_g',          'flux_auto_g',     '>f4'),
                      ('gal_fluxerr_g',       'fluxerr_auto_g',  '>f4'),
                      ('gal_mag_g',           'mag_auto_g',      '>f4'),
                      ('gal_magerr_g',        'magerr_auto_g',   '>f4'),
                      ('gal_flux_r',          'flux_auto_r',     '>f4'),
                      ('gal_fluxerr_r',       'fluxerr_auto_r',  '>f4'),
                      ('gal_mag_r',           'mag_auto_r',      '>f4'),
                      ('gal_magerr_r',        'magerr_auto_r',   '>f4'),
                      ('gal_flux_i',          'flux_auto_i',     '>f4'),
                      ('gal_fluxerr_i',       'fluxerr_auto_i',  '>f4'),
                      ('gal_mag_i',           'mag_auto_i',      '>f4'),
                      ('gal_magerr_i',        'magerr_auto_i',   '>f4'),
                      ('gal_flux_z',          'flux_auto_z',     '>f4'),
                      ('gal_fluxerr_z',       'fluxerr_auto_z',  '>f4'),
                      ('gal_mag_z',           'mag_auto_z',      '>f4'),
                      ('gal_magerr_z',        'magerr_auto_z',   '>f4'),
                      ('gal_fluxiso_u',       'flux_iso_u',      '>f4'),
                      ('gal_fluxisoerr_u',    'fluxerr_iso_u',   '>f4'),
                      ('gal_magiso_u',        'mag_iso_u',       '>f4'),
                      ('gal_magisoerr_u',     'magerr_iso_u',    '>f4'),
                      ('gal_fluxiso_g',       'flux_iso_g',      '>f4'),
                      ('gal_fluxisoerr_g',    'fluxerr_iso_g',   '>f4'),
                      ('gal_magiso_g',        'mag_iso_g',       '>f4'),
                      ('gal_magisoerr_g',     'magerr_iso_g',    '>f4'),
                      ('gal_fluxiso_r',       'flux_iso_r',      '>f4'),
                      ('gal_fluxisoerr_r',    'fluxerr_iso_r',   '>f4'),
                      ('gal_magiso_r',        'mag_iso_r',       '>f4'),
                      ('gal_magisoerr_r',     'magerr_iso_r',    '>f4'),
                      ('gal_fluxiso_i',       'flux_iso_i',      '>f4'),
                      ('gal_fluxisoerr_i',    'fluxerr_iso_i',   '>f4'),
                      ('gal_magiso_i',        'mag_iso_i',       '>f4'),
                      ('gal_magisoerr_i',     'magerr_iso_i',    '>f4'),
                      ('gal_fluxiso_z',       'flux_iso_z',      '>f4'),
                      ('gal_fluxisoerr_z',    'fluxerr_iso_z',   '>f4'),
                      ('gal_magiso_z',        'mag_iso_z',       '>f4'),
                      ('gal_magisoerr_z',     'magerr_iso_z',    '>f4'),
                      ('gal_extinct_ebv',     'extinct_ebv',     '>f4'),
                      ('gal_extinct_av',      'extinct_av',      '>f4')]

    gal_phys_dtype = [('gal_logMst',        'logMst',         '<f4'),
                      ('gal_logMst_l68',    'logMst_l68',     '<f4'),
                      ('gal_logMst_u68',    'logMst_u68',     '<f4'),
                      ('gal_logage',        'logage',         '<f4'),
                      ('gal_logage_l68',    'logage_l68',     '<f4'),
                      ('gal_logage_u68',    'logage_u68',     '<f4'),
                      ('gal_logtau',        'logtau',         '<f4'),
                      ('gal_logtau_l68',    'logtau_l68',     '<f4'),
                      ('gal_logtau_u68',    'logtau_u68',     '<f4'),
                      ('gal_logSFR',        'logSFR',         '<f4'),
                      ('gal_logSFR_l68',    'logSFR_l68',     '<f4'),
                      ('gal_logSFR_u68',    'logSFR_u68',     '<f4'),
                      ('gal_Z_star',        'Z_star',         '<f4'),
                      ('gal_Z_star_l68',    'Z_star_l68',     '<f4'),
                      ('gal_Z_star_u68',    'Z_star_u68',     '<f4'),
                      ('gal_logOH',         'logOH',          '<f4'),
                      ('gal_logOH_l68',     'logOH_l68',      '<f4'),
                      ('gal_logOH_u68',     'logOH_u68',      '<f4'),
                      ('gal_logU',          'logU',           '<f4'),
                      ('gal_logU_l68',      'logU_l68',       '<f4'),
                      ('gal_logU_u68',      'logU_u68',       '<f4'),
                      ('gal_AV',            'AV',             '<f4'),
                      ('gal_AV_l68',        'AV_l68',         '<f4'),
                      ('gal_AV_u68',        'AV_u68',         '<f4'),
                      ('gal_tauv_eff',      'tauv_eff',       '<f4'),
                      ('gal_tauv_eff_l68',  'tauv_eff_l68',   '<f4'),
                      ('gal_tauv_eff_u68',  'tauv_eff_u68',   '<f4')]

    clm_phot_dtype = [('clump_id',          'ID',             '>i4'),
                      ('clump_x',           'X',              '>f4'),
                      ('clump_y',           'Y',              '>f4'),
                      ('clump_ra',          'RA',             '>f8'),
                      ('clump_dec',         'DEC',            '>f8'),
                      ('clump_specobjid',    None,            'U20'),
                      ('clump_dist_xy',     'DISTANCE_XY',    '>f4'),
                      ('clump_distance',    'DISTANCE',       '>f4'),
                      ('clump_distnorm',    'DISTNORM',       '>f4'),
                      ('clump_dist_sma',    'DIST_SMA',       '>f4'),
                      ('clump_distphys',    'DISTPHYS',       '>f4'),
                      ('clump_psf_avg',     'PSF_WIDTH_AVG',  '>f4'),
                      ('clump_npix_aper',   'NPIX_APER',      '>f4'),
                      ('clump_prox_flag',   'PROX_FLAG',      '>i4')]

    for filt in sdss_filters:
        clm_phot_dtype += [('clump_flux_%s'%filt,       'FLUX_%s'%filt,       '>f4'),
                           ('clump_fluxerr_%s'%filt,    'FLUXERR_%s'%filt,    '>f4'),
                           ('clump_mag_%s'%filt,        'MAG_%s'%filt,        '>f4'),
                           ('clump_magerr_%s'%filt,     'MAGERR_%s'%filt,     '>f4'),
                           ('clump_orgflux_%s'%filt,    'ORGFLUX_%s'%filt,    '>f4'),
                           ('clump_orgfluxerr_%s'%filt, 'ORGFLUXERR_%s'%filt, '>f4'),
                           ('clump_orgmag_%s'%filt,     'ORGMAG_%s'%filt,     '>f4'),
                           ('clump_orgmagerr_%s'%filt,  'ORGMAGERR_%s'%filt,  '>f4'),
                           ('clump_undflux_%s'%filt,    'UNDFLUX_%s'%filt,    '>f4'),
                           ('clump_undfluxerr_%s'%filt, 'UNDFLUXERR_%s'%filt, '>f4'),
                           ('clump_undmag_%s'%filt,     'UNDMAG_%s'%filt,     '>f4'),
                           ('clump_undmagerr_%s'%filt,  'UNDMAGERR_%s'%filt,  '>f4'),
                           ('clump_difflux_%s'%filt,    'DIFFLUX_%s'%filt,    '>f4'),
                           ('clump_diffstd_%s'%filt,    'DIFFSTD_%s'%filt,    '>f4'),
                           ('clump_imgbkgd_%s'%filt,    'IMGBKGD_%s'%filt,    '>f4')]

    clm_phys_dtype = [('clump_beagle_id',     'ID',             '<i4'),
                      ('clump_logMst',        'logMst',         '<f4'),
                      ('clump_logMst_l68',    'logMst_l68',     '<f4'),
                      ('clump_logMst_u68',    'logMst_u68',     '<f4'),
                      ('clump_logage',        'logage',         '<f4'),
                      ('clump_logage_l68',    'logage_l68',     '<f4'),
                      ('clump_logage_u68',    'logage_u68',     '<f4'),
                      ('clump_logSFR',        'logSFR',         '<f4'),
                      ('clump_logSFR_l68',    'logSFR_l68',     '<f4'),
                      ('clump_logSFR_u68',    'logSFR_u68',     '<f4'),
                      ('clump_Z_star',        'Z_star',         '<f4'),
                      ('clump_Z_star_l68',    'Z_star_l68',     '<f4'),
                      ('clump_Z_star_u68',    'Z_star_u68',     '<f4'),
                      ('clump_logOH',         'logOH',          '<f4'),
                      ('clump_logOH_l68',     'logOH_l68',      '<f4'),
                      ('clump_logOH_u68',     'logOH_u68',      '<f4'),
                      ('clump_logU',          'logU',           '<f4'),
                      ('clump_logU_l68',      'logU_l68',       '<f4'),
                      ('clump_logU_u68',      'logU_u68',       '<f4'),
                      ('clump_AV',            'AV',             '<f4'),
                      ('clump_AV_l68',        'AV_l68',         '<f4'),
                      ('clump_AV_u68',        'AV_u68',         '<f4'),
                      ('clump_tauv_eff',      'tauv_eff',       '<f4'),
                      ('clump_tauv_eff_l68',  'tauv_eff_l68',   '<f4'),
                      ('clump_tauv_eff_u68',  'tauv_eff_u68',   '<f4')]

    cat_dtype = gzh_cat_dtype+gal_imag_dtype+gal_phot_dtype+gal_phys_dtype+clm_phot_dtype+clm_phys_dtype
    cat_dtype = [(new,fmt) for new,old,fmt in cat_dtype]

    dict_ = {"full"     : cat_dtype,
             "gzh_cat"  : gzh_cat_dtype,
             "gal_imag" : gal_imag_dtype,
             "gal_phot" : gal_phot_dtype,
             "gal_phys" : gal_phys_dtype,
             "clm_phot" : clm_phot_dtype,
             "clm_phys" : clm_phys_dtype}

    return dict_

from useful import *
from utils import *
from utils_DR14 import getDR14SpectraMatches

def mkClumpCatalog(sample,savename,destdir="boxcar10"):

    N = getTotalNClumps(sample=sample,destdir=destdir)
    dtype = clumpCatalogDtype()
    catalog = np.recarray(N,dtype=dtype["full"])

    gal_phot = fitsio.getdata("photom/galaxy/galaxy_photom.fits")
    gal_phys = fitsio.getdata("beagle/data/clumps_beagle_output.galiso.fits")
    clm_phys = fitsio.getdata("beagle/data/clumps_beagle_output.const.fits")

    i = 0
    for j,entry in enumerate(sample):

        print("\rProcessing objid#{0[survey_id]:020d} ({1:d}/{2:d}) ... ".format(entry,j+1,len(sample)),end="")
        photname = "photom/{1:s}/objID{0[survey_id]:020d}-phot.fits".format(entry,destdir)

        try:
            ### Clump photometry columns
            phot = fitsio.getdata(photname)
            for x,y,f in dtype["clm_phot"]:
                if y: catalog[i:i+len(phot)][x] = phot[y]

            ### GZH columns
            for x,y,f in dtype["gzh_cat"]:
                if y: catalog[i:i+len(phot)][x] = entry[y]

            ### Galaxy imaging parameters
            for x,y,f in dtype["gal_imag"]:
                if y: catalog[i:i+len(phot)][x] = phot[y]

            ### Galaxy photometry
            for x,y,f in dtype["gal_phot"]:
                if y: catalog[i:i+len(phot)][x] = gal_phot[j][y]

            ### Galaxy physical properties
            for x,y,f in dtype["gal_phys"]:
                if y: catalog[i:i+len(phot)][x] = gal_phys[j][y]

            ### All SDSS Spectro search
            spec_center = getDR14SpectraMatches(ra=phot["GAL_RA"][0], dec=phot["GAL_DEC"][0])
            if spec_center is not None:
                catalog[i:i+len(phot)]["gal_specobjid"] = spec_center["specobjid"]

            for k,clump in enumerate(phot):
                spec_match = getDR14SpectraMatches(ra=clump["RA"], dec=clump["DEC"])
                valid = spec_match is not None
                if valid and (spec_center is not None):
                    valid = spec_match["specobjid"] != spec_center["specobjid"]
                if valid:
                    catalog[i+k]["clump_specobjid"] = spec_match["specobjid"]

            i += len(phot)

        except OSError: pass

    ### Clump physical properties
    for x,y,f in dtype["clm_phys"]:
        catalog[x] = clm_phys[y]

    print("done!")

    print("Final catalog: {:d} clumps".format(len(catalog)))
    fitsio.writeto(savename, catalog, overwrite=True)

def mkTrimCatalog(catname,savename):

    rename = [("gzh_id"          ,"id"),
              ("gal_ra"          ,"ra"),
              ("gal_dec"         ,"dec"),
              ("gal_redshiftdr14","redshift"),
              ("gal_specobjid"   ,"specobjid"),
              ("gzh_run"         ,"run"),
              ("gzh_rerun"       ,"rerun"),
              ("gzh_camcol"      ,"camcol"),
              ("gzh_field"       ,"field"),
              ("gzh_obj"         ,"obj"),
              ("gzh_psf_width_u" ,"psf_fwhm_u"),
              ("gzh_psf_width_g" ,"psf_fwhm_g"),
              ("gzh_psf_width_r" ,"psf_fwhm_r"),
              ("gzh_psf_width_i" ,"psf_fwhm_i"),
              ("gzh_psf_width_z" ,"psf_fwhm_z"),
              ("gal_reff"        ,"gal_reff"),
              ("gal_sma"         ,"gal_sma"),
              ("gal_smb"         ,"gal_smb"),
              ("gal_theta"       ,"gal_theta")]
    rename = OrderedDict(rename)

    dtype = clumpCatalogDtype()
    keep = list(rename.values())
    keep+= [_[0] for _ in dtype["gal_phot"] if ("iso" not in _[0])]
    keep+= [_[0] for _ in dtype["gal_phys"] if ("logU" not in _[0]) and ("tauv_eff" not in _[0])]
    keep+= ["clump_id","clump_ra","clump_dec","clump_specobjid","clump_prox_flag",
            "clump_distance","clump_distnorm","clump_dist_sma","clump_distphys"]
    keep+= [_[0] for _ in dtype["clm_phot"] if ("clump_flux" in _[0]) or ("clump_mag" in _[0])]
    keep+= [_[0] for _ in dtype["clm_phys"] if ("logU" not in _[0]) and ("tauv_eff" not in _[0]) and ("beagle_id" not in _[0])]
    print(keep)

    catalog = np.array(fitsio.getdata(catname))
    catalog = rfn.rename_fields(catalog,rename)
    catalog = viewFields(catalog,keep)
    fitsio.writeto(savename, catalog, overwrite=True)

if __name__ == '__main__':

    sample = fitsio.getdata("samples/final_clumpy_sample.fits")

    # mkClumpCatalog(sample=sample,savename="samples/final_clump_catalog.fits")
    mkTrimCatalog(catname="samples/final_clump_catalog.fits",
                  savename="GZH_S82_Clump_Catalog.fits")

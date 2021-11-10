import os
import numpy as np
import astropy.io.fits as fitsio

def mkBeagleOutput(catalog,resdir,savename,parsSaved):

    catalog = fitsio.getdata(os.path.join(resdir,"BEAGLE-input-files",catalog))

    dtype = [("ID",int)]
    for x in parsSaved: dtype.extend([(x,float),(x+"_l68",float),(x+"_u68",float),(x+"_ML",float)])
    output = np.recarray(len(catalog), dtype=dtype)
    for x in output.dtype.names: output[x] = -99

    for i,entry in enumerate(catalog):

        filename = "{:s}/{:d}_BEAGLE_bestfit.dat".format(resdir,entry["ID"])
        if not os.path.isfile(filename):
            print(filename)
            continue

        results = np.genfromtxt(filename,dtype=[("par","U15"),("best_fit","float"),
                                                ("conf_int_l68","float"),("conf_int_u68","float"),
                                                ("error_l68","float"),("error_u68","float"),("best_fit_ML","float")])

        best_fit  = {x["par"]: x["best_fit"]  for x in results}
        error_l68 = {x["par"]: x["error_l68"] for x in results}
        error_u68 = {x["par"]: x["error_u68"] for x in results}
        best_fit_ML = {x["par"]: x["best_fit_ML"]  for x in results}

        output[i]["ID"] = entry["ID"]
        for x in parsSaved:
            output[i][x]        = best_fit[x]
            output[i][x+"_l68"] = error_l68[x]
            output[i][x+"_u68"] = error_u68[x]
            output[i][x+"_ML"]  = best_fit_ML[x]

    fitsio.writeto(savename,output,overwrite=True)

if __name__ == '__main__':

    # parsSaved = ["logMst","logage","logSFR","logSFR10","logSFR100","Z_star","logOH","logU","AV","tauv_eff"]
    parsSaved = ["logMst","logage","logtau","logSFR","logSFR10","logSFR100","Z_star","logOH","logU","AV","tauv_eff"]

    catalog = "clumps_beagle_input.diffgal.fits"
    resdir = "results/diffgal"
    savename = "data/clumps_beagle_output.diffgal.fits"

    mkBeagleOutput(catalog=catalog,resdir=resdir,savename=savename,parsSaved=parsSaved)

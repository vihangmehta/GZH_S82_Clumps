import numpy as np
import astropy.io.fits as fitsio

def main():

    filtList = ['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z']

    hduList = fitsio.HDUList([fitsio.PrimaryHDU(),])
    colList = []
    transmission = {}

    for fname in filtList:
        filt = np.genfromtxt("filters/{:s}.filt".format(fname))
        transmission[fname] = filt[filt[:,1] > 0.01*max(filt[:,1])]
        n_wl = len(transmission[fname][:,0])
        colList.append(fitsio.Column(name=fname,
                                     format="{}E".format(n_wl*2),
                                     dim='({},2)'.format(n_wl)))

    cols = fitsio.ColDefs(colList)
    hduList.append(fitsio.BinTableHDU.from_columns(cols,nrows=1,name="TRANSMISSION"))

    for fname in filtList:
        hduList[1].data[fname][0] = transmission[fname].T

    cols = fitsio.ColDefs([fitsio.Column(name="name"       ,format="40A" ),
                           fitsio.Column(name="n_wl"       ,format="I"   ),
                           fitsio.Column(name="url"        ,format="100A"),
                           fitsio.Column(name="description",format="200A"),
                           fitsio.Column(name="airmass"    ,format="E"   )])
    hduList.append(fitsio.BinTableHDU.from_columns(cols,nrows=len(filtList),name="META DATA"))

    for i,fname in enumerate(filtList):

        hduList[2].data["name"][i] = fname
        hduList[2].data["n_wl"][i] = len(transmission[fname][:,0])
        hduList[2].data["url"][i]  = ""
        hduList[2].data["description"][i] = ""
        hduList[2].data["airmass"][i] = 0.0

    hduList.writeto("filters/clumps_beagle_filters.fits",overwrite=True)

if __name__ == '__main__':

    main()

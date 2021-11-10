import os
import numpy as np
import astropy.io.fits as fitsio
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import sfd

class Gal_Extinction():

    def __init__(self):

        self.Rv = 3.1    # Av/E(B-V)

        self.Al_Av = np.genfromtxt('extinction/SDSS.extinc',usecols=[0,2],dtype=[('filt','<U8'),('Al_Av',float)])
        self.filters = list("ugriz")
        self.Alambda_Av = dict(zip(self.filters,self.Al_Av["Al_Av"]))

    def calc_EBV(self,ra,dec):

        if isinstance(ra, float): ra  = np.array([ra,])
        if isinstance(dec,float): dec = np.array([dec,])
        coords = coord.SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
        ebv = sfd.ebv(coords)
        return ebv

    def calc_Av(self,ra=None,dec=None,ebv=None):
        if ebv is None: Av = self.calc_EBV(ra,dec) * self.Rv
        else: Av = ebv * self.Rv
        return Av

    def calc_Alambda(self,filt,Av):

        factor_mag  = self.Alambda_Av[filt] * Av
        factor_flux = 10**(factor_mag/-2.5)
        return factor_mag, factor_flux

    def remove_gal_ext(self,flux,filt,ra=None,dec=None,Av=None):

        if Av is None: Av = self.calc_Av(ra=ra,dec=dec)
        factor_mag, factor_flux = self.calc_Alambda(filt,Av=Av)

        # cond = (np.abs(mag) != 99.)
        # mag[ cond] -= factor_mag[cond]

        cond = (np.abs(flux) != 99.)
        flux[cond] /= factor_flux[cond]

        return flux

    def __repr__(self):

        pprint  = "\n*** Galactic Extinction for SDSS ***"
        pprint += "%10s -- %8s [%10s]   \n" % ("Filter","Alambda","Flux_fctr")
        pprint += "".join(['-']*40)+"\n"

        for filt in self.filters:
            factor_mag, factor_flux = self.calc_Alambda(filt,Av=0.5)
            pprint += "%10s -- %8.4f [%10.4f]\n" % (filt,factor_mag,factor_flux)

        return pprint

def mk_plot():

    gal_ext = Gal_Extinction()
    catalog = fitsio.getdata('samples/final_clumpy_sample.fits')

    fig,ax = plt.subplots(1,1,figsize=(10,7),dpi=75,tight_layout=True)

    Av = gal_ext.calc_Av(ra=catalog["RA"],dec=catalog["DEC"])
    ax.hist(Av,bins=np.arange(0.,0.6,0.02),color='k',lw=0,alpha=0.4)

    ax.set_xlim(0,0.6)
    ax.set_xlabel('$A_V$',fontsize=20)

if __name__ == '__main__':

    # gal_ext = Gal_Extinction()
    # print(gal_ext)

    mk_plot()
    plt.show()

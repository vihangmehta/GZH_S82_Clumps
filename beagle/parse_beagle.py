import os
import warnings
import numpy as np
import scipy.optimize as so
import scipy.integrate as si
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from collections import OrderedDict
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.patches import Ellipse,Circle,Rectangle
from matplotlib.lines import Line2D

from useful import light, FITS_pixscale, JPEG_pixscale

from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)

pivotWL = [('sdss_u',0.3546),('sdss_g',0.4670),('sdss_r',0.6156),('sdss_i',0.7471),('sdss_z',0.8918)]
pivotWL.sort(key=lambda x:x[1])
pivotWL = OrderedDict(pivotWL)
for x in pivotWL: pivotWL[x] = pivotWL[x] * 1e4

def getVminVmax(img,sigclip=False):

    size = img.shape[0]
    _img = img[int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0)),
               int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0))]
    _img = np.ma.masked_array(_img, mask=~np.isfinite(_img))
    if sigclip:
        _img = _img[(img<np.ma.median(_img)+3*np.ma.std(_img)) & \
                    (img>np.ma.median(_img)-3*np.ma.std(_img))]
    vmin = np.ma.median(_img) - 1.0*np.ma.std(_img)
    vmax = np.ma.median(_img) + 2.0*np.ma.std(_img)
    return vmin, vmax

class BeagleOutputParser():

    def __init__(self,objid,parsSaved,parsPlotted,catname,result_dir,filter_dir,stamp_dir,jpeg_dir,phot_dir,verbose=True):

        self.catname = catname
        self.objid = objid

        self.result_dir = result_dir
        self.filter_dir = filter_dir
        self.stamp_dir  = stamp_dir
        self.phot_dir   = phot_dir
        self.jpeg_dir   = jpeg_dir

        self.verbose = verbose

        self.nbin_2d = 80
        self.nbin_1d = 200
        self.smooth_2d = 1.5
        self.smooth_1d = 1.0

        self.Zsol = 0.01524

        self.readResults()
        self.readBeagleBestFit()

        self.parsSaved = parsSaved
        self.parsPlotted = OrderedDict(parsPlotted)
        if self.verbose:
            for x in self.parsFittedNames:
                if x not in [v["colName"] for k,v in self.parsPlotted.items()]:
                    print("{} is a fitted parameter but omitting when plotting".format(x))

        self.readChains()

    def readResults(self):

        self.catalog = fitsio.getdata("{:s}/BEAGLE-input-files/{:s}".format(self.result_dir,self.catname))
        self.entry = self.catalog[self.catalog["ID"]==self.objid]

        if len(self.entry)!=1:
            raise Exception("Invalid entry for ID{:d} in {:s}".format(self.objid,self.catname))
        self.entry = self.entry[0]
        self.zspec = self.entry["zspec"]

        self.result = fitsio.open("{:s}/{:d}_BEAGLE.fits.gz".format(self.result_dir,self.objid))
        if self.zspec<1.5:
            self.spec_indices = self.result["SPECTRAL INDICES"].data.columns.names[:-6]
        elif self.zspec>=1.5:
            self.spec_indices = self.result["SPECTRAL INDICES"].data.columns.names[-6:]

    def readBeagleBestFit(self):

        self.parsFittedNames = self.result["POSTERIOR PDF"].columns.names[4:]

        ### Read in the MNstats file with the output results
        with open("{:s}/{:d}_BEAGLE_MNstats.dat".format(self.result_dir,self.objid),"r") as f:
            lines = f.readlines()

        ### Read the number of different modes
        idx = np.where(["Total Modes Found:" in line for line in lines])[0]
        if len(idx)>0:
            self.nmodes = int(lines[idx[0]].split()[-1])
        else:
            self.nmodes = 1

        ### Read the evidence for each mode and pick the best one
        self.local_logevi, self.local_logevi_strict = np.zeros((2,self.nmodes)) * np.NaN
        idx = np.where(["Strictly Local Log-Evidence" in line for line in lines])[0]
        if len(idx)>0:
            for i,_idx in enumerate(idx):
                self.local_logevi_strict[i] = float(lines[_idx].split()[-3])
                self.local_logevi[i] = float(lines[_idx+1].split()[-3])
            self.imode = np.argmax(self.local_logevi)
        else:
            self.imode = 0

        ### Read the best MAP soln
        self.parsBestFit, self.chainID = OrderedDict(), OrderedDict()
        idx = np.where(["MAP Parameters" in line for line in lines])[0]
        for i,_idx in enumerate(idx):
            theta = dict(zip(self.parsFittedNames,
                            [float(line.split()[-1]) for line in lines[_idx+2:_idx+2+len(self.parsFittedNames)]]))
            label = "MAP" if i==self.imode else "MAP_{:d}".format(i+1)
            self.parsBestFit[label] = theta
            self.chainID[label] = self.getChainID(theta)

        ### Read the best ML soln
        idx = np.where(["Maximum Likelihood Parameters" in line for line in lines])[0]
        for i,_idx in enumerate(idx):
            theta = dict(zip(self.parsFittedNames,
                            [float(line.split()[-1]) for line in lines[_idx+2:_idx+2+len(self.parsFittedNames)]]))
            label = "ML" if i==self.imode else "ML_{:d}".format(i+1)
            self.parsBestFit[label] = theta
            self.chainID[label] = self.getChainID(theta)

    def getChainID(self,theta):

        return np.argmin(sum([np.abs(self.result["POSTERIOR PDF"].data[par]-theta[par]) for par in theta]))

    def readChains(self):

        for k,v in self.parsPlotted.items():
            extName = v["extName"]
            colName = v["colName"]
            v["chain"] = self.result[extName].data[colName]
            if v["logify"]:
                v["chain"] = np.log10(v["chain"])
            if v["fix_metal_units"]:
                v["chain"] = np.log10(v["chain"] / self.Zsol)

        self.probChain = self.result["POSTERIOR PDF"].data["probability"]
        self.lnlChain = self.result["POSTERIOR PDF"].data["ln_likelihood"]

    def saveParameters(self):

        with open(os.path.join(self.result_dir,"{:d}_BEAGLE_bestfit.dat".format(self.objid)),"w") as f:

            hdrstring = "# BEAGLE best-fit parameters\n" \
                        "# ID: {:d}\n" \
                        "# zspec: {:.4f}\n" \
                        "# num of modes: {:d}\n" \
                        "# {:>13s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}\n".format(self.objid,self.zspec,self.nmodes,
                                "parameter","best_fit","conf_int_l68","conf_int_u68","error_l68","error_u68","best_fit_ML")
            f.write(hdrstring)
            print(hdrstring,end="")

            for par in self.parsSaved:
                bestMAP = self.parsPlotted[par]["chain"][self.chainID["MAP"]]
                bestML  = self.parsPlotted[par]["chain"][self.chainID["ML"]]
                pdf, binsx, confInt = self.density1D(par)
                le ,ue  = np.array(confInt[-1])-bestMAP
                parstring = "{:>15s}{:15.6e}{:15.6e}{:15.6e}{:15.6e}{:15.6e}{:15.6e}\n".format(par,bestMAP,*confInt[-1],le,ue,bestML)
                f.write(parstring)
                print(parstring,end="")

    def getPlottingColor(self,par,par2=None):

        if par2 is not None:
            condFit = [self.parsPlotted[par]["fitted"], self.parsPlotted[par2]["fitted"]]
        else:
            condFit = [self.parsPlotted[par]["fitted"]]

        if   all(condFit):
            return "tab:blue", plt.cm.Blues
        elif any(condFit):
            return "tab:orange", plt.cm.Oranges
        else:
            return "tab:green", plt.cm.Greens

    def setupTraceFigure(self,new=False):

        ndim = len(self.parsPlotted.keys())
        if new:
            self.figTrace,self.axTrace = plt.subplots(ndim+1,1,figsize=(10,(ndim+1)*2),dpi=75,sharex=True)
            self.figTrace.subplots_adjust(left=0.08,bottom=0.05,right=0.98,top=0.98,wspace=0,hspace=0.05)

        self.axTrace[-1].set_ylabel("ln(P)",fontsize=14)
        self.axTrace[-1].set_xlabel("Iteration #",fontsize=14)

        for i,par in enumerate(self.parsPlotted):
            self.axTrace[i].set_ylabel(par,fontsize=14)
            [tick.set_visible(False) for tick in self.axTrace[i].get_xticklabels()]

        for ax in self.axTrace:
            ax.minorticks_on()
            ax.tick_params(which='major', direction='in', length=5, width=1.25, right=True, top=True)
            ax.tick_params(which='minor', direction='in', length=3, width=1, right=True, top=True)

    def plotTrace(self,savefig=False):

        self.setupTraceFigure(new=True)

        self.axTrace[-1].plot(self.lnlChain, c='tab:orange', lw=1.5, alpha=1.0)
        self.axTrace[-1].set_xlim(0,len(self.lnlChain))

        for i,par in enumerate(self.parsPlotted):
            self.axTrace[i].plot(self.parsPlotted[par]["chain"], color=self.getPlottingColor(par)[0], lw=0.5, alpha=0.8)
            self.axTrace[i].axhline(self.parsPlotted[par]["chain"][self.chainID["MAP"]], c='tab:red', ls='-',  lw=1.5, alpha=0.8)
            self.axTrace[i].axhline(self.parsPlotted[par]["chain"][self.chainID["ML"]], c='tab:red', ls='--', lw=1.5, alpha=0.8)

        if savefig:
            self.figTrace.savefig(os.path.join(self.result_dir,"plots/{:d}_BEAGLE_trace.png".format(self.objid)))
            plt.close(self.figTrace)

    def setupCornerFigure(self,new=False):

        ndim = len(self.parsPlotted.keys())
        if new:
            self.figCorner,self.axCorner = plt.subplots(ndim,ndim,figsize=(12,11.5),dpi=75,sharex="col")
            self.figCorner.subplots_adjust(left=0.07,bottom=0.07,top=0.98,right=0.98,hspace=0.05,wspace=0.05)
            self.axCorner = self.axCorner.T

        for i,pari in enumerate(self.parsPlotted.keys()):
            for j,parj in enumerate(self.parsPlotted.keys()):

                if j!=ndim-1: [tick.set_visible(False) for tick in self.axCorner[i,j].get_xticklabels()]
                if i!=0:      [tick.set_visible(False) for tick in self.axCorner[i,j].get_yticklabels()]

                self.axCorner[i,-1].set_xlabel(pari,fontsize=12)
                if j>0: self.axCorner[0, j].set_ylabel(parj,fontsize=12)

                self.axCorner[i,j].minorticks_on()
                self.axCorner[i,j].tick_params(which='major', direction='in', length=5, width=1.25, right=True, top=True)
                self.axCorner[i,j].tick_params(which='minor', direction='in', length=3, width=1, right=True, top=True)

                if i==j:
                    self.axCorner[i,j].tick_params(which='both', direction='in', right=False, left=False)
                    [tick.set_visible(False) for tick in self.axCorner[i,j].get_yticklabels()]

        [self.axCorner[i,j].set_visible(False) for i in range(ndim) for j in range(ndim) if i>j]

    def findLevel(self,pdf,crit,dbinx=1,dbiny=1):

        def func(pdf,x):
            _pdf = pdf.copy()
            _pdf[_pdf<x] = 0
            return np.sum(_pdf) * dbinx * dbiny

        return so.brentq(lambda x: func(pdf,x) - crit, 0, np.max(pdf))

    def density1D(self,parx):

        xdata = self.parsPlotted[parx]["chain"]
        weights = self.probChain

        binsx = np.linspace(*self.getBinRange(xdata),self.nbin_1d)
        bincx = 0.5*(binsx[1:]+binsx[:-1])
        dbinx = np.diff(binsx)[0]

        hist = np.histogram(xdata,bins=binsx,weights=weights)[0]
        hist = gaussian_filter(hist, sigma=self.smooth_1d)
        hist = hist / si.simps(hist,bincx)

        isort = np.argsort(xdata)
        cpdf  = np.cumsum(weights[isort])
        cpdf /= cpdf[-1]
        func  = interp1d(cpdf,xdata[isort])
        confInt = [func([0.5*(1.-lev/100.), 1.-0.5*(1.-lev/100.)]) for lev in [95,68]]

        ### Direct percentiles
        # confInt = [np.percentile(xdata, [50-100*conf/2,50+100*conf/2]) for conf in [0.95,0.68]]

        ### Integrate in y -- not using
        # levels = [self.findLevel(pdf=hist,dbinx=dbinx,crit=crit) for crit in [0.95,0.68]]
        # confInt_ = [[min(bincx[hist>=level]), max(bincx[hist>=level])] for level in levels]

        return hist, bincx, confInt

    def getBinRange(self,data):

        pad = (max(data) - min(data))*0.1
        return min(data)-pad, max(data)+pad

    def density2D(self,parx,pary):

        xdata = self.parsPlotted[parx]["chain"]
        ydata = self.parsPlotted[pary]["chain"]
        weights = self.probChain

        binsx = np.linspace(*self.getBinRange(xdata),self.nbin_2d)
        binsy = np.linspace(*self.getBinRange(ydata),self.nbin_2d)
        bincx = 0.5*(binsx[1:]+binsx[:-1])
        bincy = 0.5*(binsy[1:]+binsy[:-1])
        dbinx = np.diff(binsx)[0]
        dbiny = np.diff(binsy)[0]

        hist = np.histogram2d(xdata,ydata,bins=[binsx,binsy],weights=weights)[0]
        hist = gaussian_filter(hist, sigma=self.smooth_2d)
        hist = hist / si.simps(si.simps(hist,bincx,axis=0),bincy)

        try:
            levels = [self.findLevel(pdf=hist,dbinx=dbinx,dbiny=dbiny,crit=crit) for crit in [0.95,0.68]]
        except RuntimeError:
            levels = None

        return hist, bincx, bincy, levels

    def plotCorner(self,savefig=False):

        self.setupCornerFigure(new=True)

        for i,pari in enumerate(self.parsPlotted.keys()):
            for j,parj in enumerate(self.parsPlotted.keys()):

                color, cmap = self.getPlottingColor(pari,parj)

                if i<j:

                    # self.axCorner[i,j].scatter(self.parsPlotted[pari]["chain"],self.parsPlotted[parj]["chain"],color=color,s=1,alpha=0.2)

                    for label,ls in zip(["MAP","ML"],["-","--"]):
                        idx = self.chainID[label]
                        self.axCorner[i,j].plot(self.parsPlotted[pari]["chain"][idx],self.parsPlotted[parj]["chain"][idx],marker='s',markersize=3,color='tab:red',alpha=0.9)
                        self.axCorner[i,j].axvline(self.parsPlotted[pari]["chain"][idx],color="tab:red",lw=1.5,ls=ls,alpha=1.0)
                        self.axCorner[i,j].axhline(self.parsPlotted[parj]["chain"][idx],color="tab:red",lw=1.5,ls=ls,alpha=1.0)

                    pdf, binsx, binsy, levels = self.density2D(parx=pari,pary=parj)
                    if levels is not None:
                        self.axCorner[i,j].contourf(binsx,binsy,pdf.T,cmap=cmap,levels=np.append(levels,np.max(pdf)))
                        self.axCorner[i,j].contour(binsx,binsy,pdf.T,colors=[color]*len(levels),linewidths=[0.5]*len(levels),levels=levels)

                elif i==j:

                    # self.axCorner[i,j].hist(self.parsPlotted[pari]["chain"],bins=len(self.lnlChain)//100,color=color,alpha=0.6)

                    for label,ls in zip(["MAP","ML"],["-","--"]):
                        idx = self.chainID[label]
                        self.axCorner[i,j].axvline(self.parsPlotted[pari]["chain"][idx],color="tab:red",lw=1.5,ls=ls,alpha=1.0)

                    pdf, binsx, confInt = self.density1D(parx=pari)
                    self.axCorner[i,j].plot(binsx,pdf,color=color,lw=1.5,alpha=0.9)
                    self.axCorner[i,j].axvspan(*confInt[-1],color=color,lw=0,alpha=0.3)

                    try:
                        self.axCorner[i,j].set_ylim(0,1.1*np.max(pdf))
                    except ValueError:
                        pass

        if savefig:
            self.figCorner.savefig(os.path.join(self.result_dir,"plots/{:d}_BEAGLE_corner.png".format(self.objid)))
            plt.close(self.figCorner)

    def setupSpectrumFigure(self,new=False):

        if new:
            self.figSpec,self.axSpec = plt.subplots(1,1,figsize=(20,10),dpi=75)
            self.figSpec.subplots_adjust(left=0.05,bottom=0.07,top=0.98,right=0.98,hspace=0.0)

        self.axSpec.set_xlabel('Wavelength [$\\mu$m]',fontsize=18)
        self.axSpec.set_ylabel('Flux Density [$\\mu$Jy]',fontsize=18)
        # self.axSpec.set_xscale("log")
        self.axSpec.set_yscale("log")
        self.axSpec.set_xlim(2.7e3,1.04e4)
        self.axSpec.set_ylim(1e-3,1e3)

        # self.axSpec.set_xticks(np.array([4,5,6,7,8,9,10])*1e3)
        # self.axSpec.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: "%.1f"%(x/1e4) if x/1e4%1!=0 else "%d"%(x/1e4)))
        self.axSpec.yaxis.set_minor_locator(LogLocator(base=10,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12))
        self.axSpec.tick_params(which='major', direction='in', length=10, width=1, right=True, top=True)
        self.axSpec.tick_params(which='minor', direction='in', length=5, width=1, right=True, top=True)
        if "clump_id" in self.entry:
            text = "Gal ID: {:d}\nClump ID: {:d}\nBEAGLE ID: {:d}\n$z_{{spec}}$={:.4f}".format(self.entry["gal_id"],self.entry["clump_id"],self.objid,self.zspec)
        else:
            text = "Gal ID: {:d}\nBEAGLE ID: {:d}\n$z_{{spec}}$={:.4f}".format(self.entry["gal_id"],self.objid,self.zspec)
        self.axSpec.text(0.01,0.98,text,va="top",ha="left",fontsize=18,fontweight=600,transform=self.axSpec.transAxes)

        [tick.set_fontsize(14) for tick in self.axSpec.get_xticklabels()+self.axSpec.get_yticklabels()]

    def addClumpStamps(self,fig,axis,angSize=15,daxSize=0.18):

        clump_phot = fitsio.getdata(os.path.join(self.phot_dir,"objID{0[gal_id]:020d}-phot.fits".format(self.entry)))
        clump_entry = clump_phot[clump_phot["ID"]==self.entry["clump_id"]]

        N = 5
        dax = None
        len_1arcsec = 1. / FITS_pixscale
        pixSize = angSize * len_1arcsec

        width, height = fig.get_size_inches()
        aspect = height / width
        corner = axis.get_position().corners()[-1]

        for i,filt in enumerate("ugriz"):

            dax = fig.add_axes([corner[0]-0.005-daxSize*aspect*(N-i),
                                corner[1]-0.012-daxSize,
                                daxSize*aspect*0.95,
                                daxSize*0.95])

            dax.xaxis.set_visible(False)
            dax.yaxis.set_visible(False)

            cutout = fitsio.getdata(os.path.join(self.stamp_dir,"objID{0[gal_id]:020d}-{1:s}.fits".format(self.entry,filt)))
            vmin,vmax = getVminVmax(cutout[int(clump_entry["Y"]-pixSize):int(clump_entry["Y"]+pixSize),
                                           int(clump_entry["X"]-pixSize):int(clump_entry["X"]+pixSize)])
            dax.pcolormesh(cutout,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,rasterized=True)
            dax.add_patch(Circle(xy=(clump_entry["X"]-1,clump_entry["Y"]-1),radius=2/2*len_1arcsec,facecolor='none',edgecolor='lawngreen',lw=1.2))
            dax.add_patch(Circle(xy=(clump_entry["X"]-1,clump_entry["Y"]-1),radius=3/2*len_1arcsec,facecolor='none',edgecolor='lawngreen',lw=1.2))
            # dax.text(0.5,1,"%s"%filt,color='k',fontsize=12,fontweight=600,va="bottom",ha="center",transform=dax.transAxes)

            dax.set_xlim(clump_entry["X"]-pixSize/2,clump_entry["X"]+pixSize/2)
            dax.set_ylim(clump_entry["Y"]-pixSize/2,clump_entry["Y"]+pixSize/2)
            dax.set_aspect(1.)

    def addGalaxyStamps(self,fig,axis,daxSize=0.18):

        N = 5
        dax = None

        width, height = fig.get_size_inches()
        aspect = height / width
        corner = axis.get_position().corners()[-1]

        for i,filt in enumerate("ugriz"):

            dax = fig.add_axes([corner[0]-0.005-daxSize*aspect*(N-i),
                                corner[1]-0.012-daxSize,
                                daxSize*aspect*0.95,
                                daxSize*0.95])

            dax.xaxis.set_visible(False)
            dax.yaxis.set_visible(False)

            cutout = fitsio.getdata(os.path.join(self.stamp_dir,"objID{0[gal_id]:020d}-{1:s}.fits".format(self.entry,filt)))
            vmin,vmax = getVminVmax(cutout[int(cutout.shape[1]*1/3):int(cutout.shape[0]*2/3),
                                           int(cutout.shape[0]*1/3):int(cutout.shape[0]*2/3)])
            dax.pcolormesh(cutout,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,rasterized=True)
            # dax.text(0.5,1,"%s"%filt,color='k',fontsize=12,fontweight=600,va="bottom",ha="center",transform=dax.transAxes)

            dax.set_xlim(cutout.shape[0]*1/3,cutout.shape[0]*2/3)
            dax.set_ylim(cutout.shape[0]*1/3,cutout.shape[0]*2/3)
            dax.set_aspect(1.)

    def plotSpectrum(self,savefig=False):

        self.setupSpectrumFigure(new=True)

        wphot     = np.array([pivotWL[x] for x in pivotWL.keys()])
        ophot     = np.array([self.entry["flux_{:s}".format(filt)] for filt in pivotWL.keys()])
        ophot_err = np.array([self.entry["fluxerr_{:s}".format(filt)] for filt in pivotWL.keys()])

        self.axSpec.errorbar(wphot, ophot, yerr=ophot_err, label="Obs phot",
                             ecolor='k', marker='o',markersize=10, ls='', lw=3, alpha=0.9,
                             markerfacecolor='none',markeredgecolor='k', markeredgewidth=3)

        mags = ophot[ophot>0]
        for i,(label,ls,lw,marker) in enumerate(zip(["MAP","ML"],['-','--'],[1.5,0.8],['s','d'])):

            chainID = self.chainID[label]

            wspec = self.result["FULL SED WL"].data[0][0] * (1+self.result["POSTERIOR PDF"].data["redshift"][chainID])
            mspec = self.result["FULL SED"].data[chainID,:] / (1+self.result["POSTERIOR PDF"].data["redshift"][chainID])
            mphot = np.array([self.result["APPARENT MAGNITUDES"].data[x+"_APP"][chainID] for x in pivotWL.keys()])

            mspec = mspec * wspec**2 / light / 1e-23 * 1e6
            mphot = 10**((mphot+48.6)/-2.5) / 1e-23 * 1e6
            mags = np.concatenate([mags,mphot])

            self.axSpec.plot(wspec, mspec, lw=lw, ls=ls, color='tab:red', alpha=0.8)
            self.axSpec.errorbar(wphot[ophot!=-99], mphot[ophot!=-99],
                                 marker=marker, markersize=10, alpha=0.8, ls='', lw=3,
                                 markerfacecolor='none', markeredgecolor='tab:red', markeredgewidth=lw*2)

        ymin, ymax = mags.min()/3, mags.max()*12

        for i,f in enumerate(pivotWL.keys()):
            if ophot[i] != -99.:
                w, t = np.genfromtxt("{:s}/{:s}.filt".format(self.filter_dir,f),unpack=True)
                t = t / t.max()
                t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
                self.axSpec.plot(w, t, lw=3, color='gray', alpha=0.7)

        idx = self.chainID["MAP"]
        text = "log M$^\\star$: {:.2f} M$_\\odot$\n".format( self.parsPlotted["logMst"]["chain"][idx])+\
               "log Age: {:.2f} yr\n".format(                self.parsPlotted["logage"]["chain"][idx])+\
               "sSFR: {:.2f} yr$^{{-1}}$\n".format(          self.parsPlotted["logSFR"]["chain"][idx]-self.parsPlotted["logMst"]["chain"][idx])+\
               "log(Z$_\\star$/Z$_\\odot$): {:.3f}\n".format(self.parsPlotted["Z_star"]["chain"][idx])+\
               "log(OH)+12: {:.3f}\n".format(                self.parsPlotted["logOH"]["chain"][idx])+\
               "A$_V$: {:.2f}\n".format(                     self.parsPlotted["AV"]["chain"][idx])
        self.axSpec.text(0.99,0.01,text,va="bottom",ha="right",fontsize=18,fontweight=600,transform=self.axSpec.transAxes)
        self.axSpec.set_ylim([ymin, ymax])

        if "clump_id" in self.entry:
            self.addClumpStamps(fig=self.figSpec,axis=self.axSpec)
        else:
            self.addGalaxyStamps(fig=self.figSpec,axis=self.axSpec)

        if savefig:
            self.figSpec.savefig(os.path.join(self.result_dir,"plots/{:d}_BEAGLE_spectrum.png".format(self.objid)))
            plt.close(self.figSpec)

if __name__ == '__main__':

    # parsSaved = ["logMst","logage","logSFR","logSFR10","logSFR100","Z_star","logOH","logU","AV","tauv_eff"]
    parsSaved = ["logMst","logage","logtau","logSFR","logSFR10","logSFR100","Z_star","logOH","logU","AV","tauv_eff"]

    parsPlotted = [
                   ("logage"      ,{"extName":"POSTERIOR PDF"    , "colName":"max_stellar_age" , "fitted":True , "logify":False , "fix_metal_units":False}),
                   ("logtau"      ,{"extName":"POSTERIOR PDF"    , "colName":"tau"             , "fitted":True , "logify":False , "fix_metal_units":False}),
                   # ("logage"      ,{"extName":"POSTERIOR PDF"    , "colName":"ssp_age"          , "fitted":True , "logify":False , "fix_metal_units":False}),
                   ("mass"        ,{"extName":"POSTERIOR PDF"    , "colName":"mass"            , "fitted":True , "logify":False , "fix_metal_units":False}),
                   ("Z_star"      ,{"extName":"POSTERIOR PDF"    , "colName":"metallicity"     , "fitted":True , "logify":False , "fix_metal_units":False}),
                   ("tauv_eff"    ,{"extName":"POSTERIOR PDF"    , "colName":"tauv_eff"        , "fitted":True , "logify":False , "fix_metal_units":False}),
                   # ("Z_ISM"       ,{"extName":"POSTERIOR PDF"    , "colName":"nebular_Z"       , "fitted":True , "logify":False , "fix_metal_units":False}),
                   # ("logU"        ,{"extName":"POSTERIOR PDF"    , "colName":"nebular_logU"    , "fitted":True , "logify":False , "fix_metal_units":False}),
                   ("Z_ISM"       ,{"extName":"HII EMISSION"     , "colName":"Z_ISM"           , "fitted":False, "logify":False , "fix_metal_units":False}),
                   ("logU"        ,{"extName":"HII EMISSION"     , "colName":"logU"            , "fitted":False, "logify":False , "fix_metal_units":False}),
                   ("logMst"      ,{"extName":"GALAXY PROPERTIES", "colName":"M_star"          , "fitted":False, "logify":True  , "fix_metal_units":False}),
                   ("logSFR"      ,{"extName":"STAR FORMATION"   , "colName":"SFR"             , "fitted":False, "logify":True  , "fix_metal_units":False}),
                   ("logSFR10"    ,{"extName":"STAR FORMATION"   , "colName":"SFR_10"          , "fitted":False, "logify":True  , "fix_metal_units":False}),
                   ("logSFR100"   ,{"extName":"STAR FORMATION"   , "colName":"SFR_100"         , "fitted":False, "logify":True  , "fix_metal_units":False}),
                   # ("Z_star"      ,{"extName":"GALAXY PROPERTIES", "colName":"mass_w_Z"        , "fitted":False, "logify":False , "fix_metal_units":True }),
                   ("AV"          ,{"extName":"DUST ATTENUATION" , "colName":"A_V"             , "fitted":False, "logify":False , "fix_metal_units":False}),
                   ("AV_star"     ,{"extName":"DUST ATTENUATION" , "colName":"A_V_stellar"     , "fitted":False, "logify":False , "fix_metal_units":False}),
                   ("logOH"       ,{"extName":"HII EMISSION"     , "colName":"logOH"           , "fitted":False, "logify":False , "fix_metal_units":False}),
                   ]

    savefig = True
    catname = "clumps_beagle_input.diffgal.fits"
    result_dir = "results/diffgal"
    catalog = fitsio.getdata(os.path.join(result_dir,"BEAGLE-input-files",catname))

    for i,entry in enumerate(catalog):

        filename = os.path.join(result_dir,"{:d}_BEAGLE.fits.gz".format(entry["ID"]))
        if not os.path.isfile(filename): continue

        print("Plotting {:d} ({:d}/{:d}) ... ".format(entry["ID"],i+1,len(catalog)))

        plotter = BeagleOutputParser(objid=entry["ID"],
                                     parsSaved=parsSaved,
                                     parsPlotted=OrderedDict(parsPlotted),
                                     catname=catname,
                                     result_dir=result_dir,
                                     stamp_dir="../S82/fits/",
                                     jpeg_dir="../S82/jpegs/",
                                     phot_dir="../photom/boxcar10/",
                                     filter_dir="filters",
                                     verbose=False)

        plotter.saveParameters()
        # plotter.plotTrace(savefig=savefig)
        # plotter.plotCorner(savefig=savefig)
        # plotter.plotSpectrum(savefig=savefig)

    print()
    plt.show()

from useful import *
from utils import *
from utils_clumps import *

def getVminVmax(img,sigclip=False):

    size = img.shape[0]
    _img = img[int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0)),
               int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0))]
    _img = np.ma.masked_array(_img, mask=~np.isfinite(_img))
    if sigclip:
        _img = _img[(_img<np.ma.median(_img)+3*np.ma.std(_img)) & \
                    (_img>np.ma.median(_img)-3*np.ma.std(_img))]
    vmin = np.ma.median(_img) - 1.0*np.ma.std(_img)
    vmax = np.ma.median(_img) + 2.0*np.ma.std(_img)
    return vmin, vmax

def reprocessSegMap(segm):
    """
    Reprocesses a segementation map for easier plotting
    """
    iobj = segm[int(segm.shape[0] / 2), int(segm.shape[1] / 2)]
    uniq = np.sort(np.unique(segm))
    if iobj == 0:
        iobj = np.max(uniq) + 1
    uniq = uniq[(uniq != iobj)]
    uniq = np.append(uniq, iobj)
    renumber_dict = dict(zip(uniq, np.arange(len(uniq)) + 1))
    _segm = segm.copy()
    for _i in uniq:
        _segm[segm == _i] = renumber_dict[_i]
    return _segm

def plotSDSSSummary(entry,savefig=True):

    if os.path.isfile("DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"])):
        entry_dr14 = fitsio.getdata("DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"]))[0]
    else:
        entry_dr14 = {"RA":np.NaN,"DEC":np.NaN,"distance":np.NaN,
                      "redshift":np.NaN,"redshift_err":np.NaN,"petrorad_r":np.NaN,
                      "u":np.NaN,"g":np.NaN,"r":np.NaN,"i":np.NaN,"z":np.NaN,
                      "Err_u":np.NaN,"Err_g":np.NaN,"Err_r":np.NaN,"Err_i":np.NaN,"Err_z":np.NaN,}

    fig = plt.figure(figsize=(20,9),dpi=75)
    fig.subplots_adjust(left=0.01,right=0.99,bottom=0.02,top=0.98)

    ogs = gridspec.GridSpec(2,4,wspace=0.03,hspace=0.03,width_ratios=[0.2,0.125,0.05,1],height_ratios=[1,1])

    igs = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=ogs[:,0],wspace=0.05,hspace=0.05)
    ax1 = fig.add_subplot(igs[0,0])
    ax2 = fig.add_subplot(igs[1,0])
    ax3 = fig.add_subplot(igs[2,0])

    igs = gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=ogs[:,1],wspace=0.05,hspace=0.05)

    specax = fig.add_subplot(ogs[0,3])
    specax_color = cycle(plt.cm.Set1(np.linspace(0,1,10)))

    vmin,vmax = None,None
    for i,filt in enumerate(sdss_filters[::-1]):

        if os.path.isfile("S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt)):

            ax = fig.add_subplot(igs[len(sdss_filters)-i-1])
            ax.set_aspect(1.)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            img = fitsio.getdata("S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt))

            vmin,vmax = getVminVmax(img)
            ax.imshow(img,origin="lower",cmap=plt.cm.Greys_r,vmin=vmin,vmax=vmax)

    if os.path.isfile("/data/extragal/willett/gzh/jpg/{:d}.jpg".format(entry["survey_id_single"])):
        img = mpimg.imread("/data/extragal/willett/gzh/jpg/{:d}.jpg".format(entry["survey_id_single"]))
        ax1.imshow(img)
        ax1.scatter(img.shape[0]/2.,img.shape[1]/2.,c="w",marker="+",lw=0.8,s=100,alpha=0.8)

    img = mpimg.imread("S82/jpegs/objID{0[survey_id]:020d}.jpg".format(entry))
    ax2.imshow(img)
    ax2.scatter(img.shape[0]/2.,img.shape[1]/2.,c="w",marker="+",lw=0.8,s=100,alpha=0.8)

    img = mpimg.imread("DR14/jpegs/objID{0[survey_id]:020d}.jpg".format(entry))
    angSize = img.shape[0] * JPEG_pixscale
    ax3.imshow(img)
    ax3.scatter(*transformImgToWCS(entry["RA"],entry["DEC"],entry["RA"],entry["DEC"],size=angSize,scale=JPEG_pixscale),
                    c='w',marker='+',s=100,lw=0.8,alpha=0.8)
    ax3.scatter(*transformImgToWCS(entry_dr14["RA"],entry_dr14["DEC"],entry["RA"],entry["DEC"],size=angSize,scale=JPEG_pixscale),
                    c='r',marker='+',s=100,lw=0.8,alpha=0.8)
    ax3.set_xlim(0,img.shape[0])
    ax3.set_ylim(img.shape[1],0)

    use_z = entry["REDSHIFT"]
    nearbyObjectsFile = "DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"])
    if not np.isfinite(use_z) and os.path.isfile(nearbyObjectsFile):
        nearbyObjects = fitsio.getdata(nearbyObjectsFile)
        nearbyObject_z = nearbyObjects["redshift"][0]
        if np.isfinite(nearbyObject_z):
            use_z = nearbyObject_z
    ax3.add_patch(Circle(transformImgToWCS(entry["RA"],entry["DEC"],entry["RA"],entry["DEC"],size=angSize,scale=JPEG_pixscale),
                            radius=getAngularSize(z=use_z,physSize=15,angSize=10)/JPEG_pixscale,
                            facecolor="none",edgecolor="w",lw=0.8,ls='--',alpha=0.5))

    barx0,bary0,barh = img.shape[0]*0.98, img.shape[1]*0.92, img.shape[1]*0.005
    ax3.add_patch(Rectangle(xy=(barx0-5/JPEG_pixscale,bary0),width=5/JPEG_pixscale,height=barh,facecolor='w',edgecolor='none',lw=0,alpha=0.9))
    ax3.text(barx0-5/JPEG_pixscale/2, bary0+1.1*barh, "5\"", va="top", ha="center", fontsize=13, color="w")

    metadata = r"\begin{tabular}{lll}" + \
               r"& \textbf{\huge GZH} & \textbf{\huge DR14} \\" + \
               r"\hline " + \
               r"GZH ID: & \multicolumn{2}{c}{" + "{0[survey_id]:d}".format(entry,entry_dr14) + "} \\\\" + \
               r"\hline " + \
               r"$\alpha$ & " + "{0[RA]:.8f} & {1[RA]:.8f} \\\\".format(entry,entry_dr14) + \
               r"$\delta$ & " + "{0[DEC]:.8f} & {1[DEC]:.8f} \\\\".format(entry,entry_dr14) + \
               r"Dist. & $-$ & " + "{1[distance]:.4f}\" \\\\".format(entry,entry_dr14) + \
               r"\hline " + \
               r"redshift & " + "{0[REDSHIFT]:.3f}$\pm${0[REDSHIFTERR]:.3f} & {1[redshift]:.3f}$\pm${1[redshift_err]:.3f} \\\\".format(entry,entry_dr14) + \
               r"\hline " + \
               r"$u$ & " + "{0[PETROMAG_U]:.2f}$\pm${0[PETROMAGERR_U]:.2f} & {1[u]:.2f}$\pm${1[Err_u]:.2f} \\\\".format(entry,entry_dr14) + \
               r"$g$ & " + "{0[PETROMAG_G]:.2f}$\pm${0[PETROMAGERR_G]:.2f} & {1[g]:.2f}$\pm${1[Err_g]:.2f} \\\\".format(entry,entry_dr14) + \
               r"$r$ & " + "{0[PETROMAG_R]:.2f}$\pm${0[PETROMAGERR_R]:.2f} & {1[r]:.2f}$\pm${1[Err_r]:.2f} \\\\".format(entry,entry_dr14) + \
               r"$i$ & " + "{0[PETROMAG_I]:.2f}$\pm${0[PETROMAGERR_I]:.2f} & {1[i]:.2f}$\pm${1[Err_i]:.2f} \\\\".format(entry,entry_dr14) + \
               r"$z$ & " + "{0[PETROMAG_Z]:.2f}$\pm${0[PETROMAGERR_Z]:.2f} & {1[z]:.2f}$\pm${1[Err_z]:.2f} \\\\".format(entry,entry_dr14) + \
               r"\hline " + \
               r"$r_{50}$ & " + "{0[PETROR50_R]:.3f}\" ".format(entry) + r"& $-$ \\" + \
               r"$r_{90}$ & " + "{0[PETROR90_R]:.3f}\" & {1[petrorad_r]:.3f}\" \\\\".format(entry,entry_dr14) + \
               r"\hline " + \
               r"\end{tabular}"

    if os.path.isfile("DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"])):

        nearbyObjects = fitsio.getdata("DR14/photObj/nearby_photObj_ID{:020d}.fits".format(entry["survey_id"]))

        nearbyObjectsText = r"\textbf{\huge Nearby Objects (DR14)} \\ " + \
                            r"\begin{tabular}{clllllll} " + \
                            r"\hline " + \
                            r"\# & objId & RA & DEC & Dist. & $m_{r}$ & specObjId & $z$ \\ " + \
                            r"\hline "

        for i,nearbyObject in enumerate(nearbyObjects):

            _xpos,_ypos = transformImgToWCS(nearbyObject["ra"],nearbyObject["dec"],entry["RA"],entry["DEC"],size=angSize,scale=JPEG_pixscale)
            if 0<=_xpos<=img.shape[0] and 0<=_ypos<=img.shape[0]:
                ax3.text(_xpos,_ypos,i+1,va="center",ha="center",fontsize=12,color='w')

            if i<10:
                nearbyObjectsText += " {0:d} & {1[objId]:} & {1[ra]:.8f} & {1[dec]:.8f} & {1[distance]:.4f}\" & {1[r]:.2f} & {1[specObjId]:s} & {1[redshift]:.3f} \\\\".format(i+1,nearbyObject)

            if nearbyObject["specobjid"]!="":

                spec_data = fitsio.getdata("DR14/spectra/specID{:s}.fits".format(nearbyObject["specobjid"].zfill(20)),1)
                wave, spec = 10**spec_data["loglam"], spec_data["flux"]
                specax.plot(wave,spec,c=next(specax_color),lw=0.8,alpha=0.8,label=i+1)
                specax.legend(fontsize=14)

        nearbyObjectsText += r" \hline " + \
                             r" \end{tabular} "

    else:

        nearbyObjectsText = r"{\huge No matches in DR14}"

    nearbyObjectsText = nearbyObjectsText.replace("9223372036854775808","nan")
    matplotlib.rc('text', usetex=True)
    specax.text(0.28,0.42, metadata,  va="top",ha="left",fontsize=15,fontweight=500,transform=fig.transFigure)
    specax.text(0.50,0.42, nearbyObjectsText, va="top",ha="left",fontsize=15,fontweight=500,transform=fig.transFigure)
    matplotlib.rc('text', usetex=False)

    for ax in [ax1,ax2,ax3]:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_aspect(1.)

    for pos in ["left","right","top","bottom"]:
        if   entry["clumpy_selection"]=="single":
            ax1.spines[pos].set_edgecolor("tab:red")
            ax1.spines[pos].set_linewidth(3)
        elif entry["clumpy_selection"]=="coadd":
            ax2.spines[pos].set_edgecolor("tab:red")
            ax2.spines[pos].set_linewidth(3)

    specax.set_xlim(3800,9200)
    specax.axhline(0,c='k',ls='--',lw=0.8,alpha=0.8)
    specax.set_ylabel("Flux [$\\times$ 10$^{-17}$ cgs$\\AA$]",fontsize=16)
    specax.set_xlabel("Observed Wavelength [$\\AA$]",fontsize=16)
    [_.set_fontsize(14) for _ in specax.get_xticklabels()+specax.get_yticklabels()]

    if savefig:
        fig.savefig("plots/summary_sdss/objID{:020d}.png".format(entry["survey_id"]))
        plt.close(fig)
    else:
        return fig

def plotClumpsDetection(entry,savefig=True):

    figsize = np.array([13,6.55]) * 1.5
    fig = plt.figure(figsize=figsize, dpi=75)
    fig.subplots_adjust( left=  0.05/figsize[0]*figsize[1],
                        right=1-0.05/figsize[0]*figsize[1],
                        bottom=0.05,top=0.92)
    fig.suptitle("ID {:d}".format(entry["survey_id"]),fontsize=22,fontweight=600)

    ogs = fig.add_gridspec(1,5,width_ratios=[1,3,3,3,3],wspace=0.1,hspace=0.1)

    igs0 = ogs[0].subgridspec(6,1,wspace=0,hspace=0,height_ratios=[2,2,2,2,2,3])
    igs1 = ogs[1].subgridspec(6,6,wspace=0,hspace=0,height_ratios=[2,2,2,2,2,3])
    igs2 = ogs[2].subgridspec(6,6,wspace=0,hspace=0,height_ratios=[2,2,2,2,2,3])
    igs3 = ogs[3].subgridspec(6,6,wspace=0,hspace=0,height_ratios=[2,2,2,2,2,3])
    igs4 = ogs[4].subgridspec(6,6,wspace=0,hspace=0,height_ratios=[2,2,2,2,2,3])

    for i, filt in enumerate(sdss_filters):

        img = fitsio.getdata("S82/fits/objID{:020d}-{:s}.fits".format(entry["survey_id"],filt))
        vmin, vmax = getVminVmax(img)

        ax = fig.add_subplot(igs0[i])
        ax.imshow(img,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,origin="lower",rasterized=True)
        ax.set_xlim(img.shape[0]*0.05,img.shape[0]*0.95)
        ax.set_ylim(img.shape[1]*0.05,img.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if i==0: ax.set_title("SCI", fontsize=16, fontweight=600)

    for _igs,destdir in zip([igs1,igs2,igs3,igs4],["gauss10","gauss15","boxcar10","boxcar15"]):

        for i,filt in enumerate(sdss_filters):

            for j,label in enumerate(["smooth","cntrst","filter"]):

                img = fitsio.getdata("photom/{3:s}/objID{0:020d}-{1:s}.{2:s}.fits".format(entry["survey_id"],filt,label,destdir))
                vmin, vmax = getVminVmax(img)

                ax = fig.add_subplot(_igs[i,2*j:2*j+2])
                ax.imshow(img,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,origin="lower",rasterized=True)
                ax.set_xlim(img.shape[0]*0.05,img.shape[0]*0.95)
                ax.set_ylim(img.shape[1]*0.05,img.shape[1]*0.95)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

                if i==0 and j==1: ax.set_title(destdir, fontsize=14, fontweight=600)

        img = fitsio.getdata("photom/{1:s}/objID{0:020d}-det.fits".format(entry["survey_id"], destdir))
        vmin, vmax = getVminVmax(img)
        ax = fig.add_subplot(_igs[i+1,:3])
        ax.imshow(img,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax,origin="lower")
        ax.set_xlim(img.shape[0]*0.05,img.shape[0]*0.95)
        ax.set_ylim(img.shape[1]*0.05,img.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        segm = fitsio.getdata("photom/{1:s}/objID{0:020d}-seg.fits".format(entry["survey_id"], destdir))
        segm = reprocessSegMap(segm)
        ax = fig.add_subplot(_igs[i+1,3:])
        ax.imshow(segm,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segm),origin="lower")
        ax.set_xlim(img.shape[0]*0.05,img.shape[0]*0.95)
        ax.set_ylim(img.shape[1]*0.05,img.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if savefig:
        fig.savefig("plots/summary_clumps/objID{:020d}-clumps.png".format(entry["survey_id"]))
        plt.close(fig)
    else:
        return fig

class InteractiveClumpCurator():

    def __init__(self,axes,transform,fullcat,trimcat):

        self.axis_dr14 = axes[0]
        self.axis_addn = axes[1]
        self.axis      = axes[2]
        self.transform = transform

        self.keepPts   = None
        self.rejectPts = None
        self.newRejPts = None
        self.otherPts  = None
        self.picker = 3

        self.fullCat = fullcat
        self.trimCat = trimcat
        self.newTrim = trimcat
        self.rejectList = np.zeros((2,0))
        self.rejCat  = self.fullCat[~np.in1d(self.fullCat["NUMBER"],self.trimCat["ID"])]

        self.cid = self.axis.figure.canvas.mpl_connect('pick_event', self)
        self.drawCanvas()

    def drawCanvas(self):

        if self.keepPts:   self.keepPts.remove()
        if self.rejectPts: self.rejectPts.remove()
        if self.newRejPts: self.newRejPts.remove()
        if self.otherPts:  [otherPts.remove() for otherPts in self.otherPts]

        self.newRejCat = self.trimCat[~np.in1d(self.trimCat["ID"],self.newTrim["ID"])]

        self.rejectPts = self.axis.scatter(self.rejCat["X_IMAGE"]-1,self.rejCat["Y_IMAGE"]-1,s=30,color="magenta",marker="o",lw=0,picker=self.picker)
        otherPts1 = self.axis_addn.scatter(self.rejCat["X_IMAGE"]-1,self.rejCat["Y_IMAGE"]-1,s=30,color="magenta",marker="x",lw=1)
        otherPts2 = self.axis_dr14.scatter(*self.transform(self.rejCat["X_WORLD"],self.rejCat["Y_WORLD"]),s=30,color="magenta",marker="x",lw=1)

        self.keepPts   = self.axis.scatter(self.newTrim["X"]-1,self.newTrim["Y"]-1,s=30,color="lawngreen",marker="o",lw=0,picker=self.picker)
        otherPts3 = self.axis_addn.scatter(self.newTrim["X"]-1,self.newTrim["Y"]-1,s=30,color="lawngreen",marker="x",lw=1)
        otherPts4 = self.axis_dr14.scatter(*self.transform(self.newTrim["RA"],self.newTrim["DEC"]),s=30,color="lawngreen",marker="x",lw=1)

        self.newRejPts = self.axis.scatter(self.newRejCat["X"]-1,self.newRejCat["Y"]-1,s=30,color="r",marker="o",lw=0,picker=self.picker)
        otherPts5 = self.axis_addn.scatter(self.newRejCat["X"]-1,self.newRejCat["Y"]-1,s=30,color="r",marker="x",lw=1)
        otherPts6 = self.axis_dr14.scatter(*self.transform(self.newRejCat["RA"],self.newRejCat["DEC"]),s=30,color="r",marker="x",lw=1)

        self.otherPts = [otherPts1,otherPts2,otherPts3,otherPts4,otherPts5,otherPts6]

        self.axis.figure.canvas.draw()
        self.axis_addn.figure.canvas.draw()
        self.axis_dr14.figure.canvas.draw()

    def __call__(self,event):

        idx = np.atleast_1d(event.ind)

        if len(idx)==0:
            return True

        if len(idx)>1:
            print("Multiple markers within click radius, try again")
            return True

        idx = idx[0]
        if event.artist==self.rejectPts:
            print("Clump is outside the galaxy segmap (already rejected)")

        elif event.artist==self.keepPts:
            marked = self.newTrim[idx]
            xpos,ypos = np.round([marked["X"],marked["Y"]],2)
            print("Rejecting clump#{:d} at location [{:.2f},{:.2f}]".format(marked["ID"],xpos,ypos))
            self.rejectList = np.append(self.rejectList,[[xpos],[ypos]],axis=-1)
            self.newTrim = np.delete(self.newTrim,idx)
            self.drawCanvas()

        elif event.artist==self.newRejPts:
            marked = self.newRejCat[idx]
            xpos,ypos = np.round([marked["X"],marked["Y"]],2)
            print("Re-adding clump#{:d} at location [{:.2f},{:.2f}]".format(marked["ID"],xpos,ypos))
            idx = np.where((self.rejectList[0,:]==xpos) & (self.rejectList[1,:]==ypos))[0]
            self.rejectList = np.delete(self.rejectList,idx,axis=-1)
            self.newTrim = rfn.stack_arrays([self.newTrim,marked],usemask=False,asrecarray=True)
            self.newTrim.sort(order="ID")
            self.drawCanvas()

        return True

def plotClumpsCuration(entry,destdir,trimcat=None,markPts=True,savefig=True):

    fullcat = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-cat.fits".format(entry, destdir))
    if trimcat is None:
        try:
            trimcat = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-trim.fits".format(entry, destdir))
        except OSError:
            trimcat = None
    elif len(trimcat)==0:
        trimcat = None

    fig = plt.figure(figsize=(13.8,12),dpi=75)
    fig.subplots_adjust(left=0,right=1,top=0.95,bottom=0,wspace=0.1,hspace=0.1)
    fig.suptitle("ID {:d} ({:s})".format(entry["survey_id"],destdir),fontsize=22,fontweight=600)

    ogs = fig.add_gridspec(2,3,width_ratios=[5,5,2],wspace=0,hspace=0)
    ax1 = fig.add_subplot(ogs[0,0])
    ax2 = fig.add_subplot(ogs[1,0])
    ax3 = fig.add_subplot(ogs[0,1])
    ax4 = fig.add_subplot(ogs[1,1])

    igs = ogs[:,-1].subgridspec(5,1,wspace=0,hspace=0)
    axes = [fig.add_subplot(igs[j]) for j in range(5)]

    segimg = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-seg.fits".format(entry,destdir))
    segimg = reprocessSegMap(segimg)
    ax3.imshow(segimg,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segimg),origin="lower")

    segimg = fitsio.getdata("photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry))
    segimg = reprocessSegMap(segimg)
    ax4.imshow(segimg,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segimg),origin="lower")

    apersize = getClumpApertureSize(entry)
    stmask = getStarMask(entry,imshape=segimg.shape,radius=0.5*apersize["star"])
    stmask = np.ma.masked_array(stmask,mask=stmask==0)
    for ax in [ax3,ax4]:
        ax.imshow(stmask,cmap=plt.cm.cool_r,vmin=0,vmax=1,origin="lower")

    for j,filt in enumerate(sdss_filters):

        img,img_hdr = fitsio.getdata("S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt),header=True)
        vmin,vmax = getVminVmax(img)
        axes[j].imshow(img,origin="lower",cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)

    for ax in [ax3,ax4]:

        if markPts:
            rejcat = fullcat[~np.in1d(fullcat["NUMBER"],trimcat["ID"])]
            ax.scatter(rejcat["X_IMAGE"]-1,rejcat["Y_IMAGE"]-1,s=50,color="r",marker="x",lw=2)
            if trimcat is not None:
                ax.scatter(trimcat["X"]-1,trimcat["Y"]-1,s=50,color="lawngreen",marker="x",lw=2)

        if trimcat is not None:
            # Mark the computed CoM for the diffuse light
            ax.scatter(trimcat["GAL_XC"][0]-1,trimcat["GAL_YC"][0]-1,s=250,color="lawngreen",marker="+",lw=3)
            # Add a circle showing the petro radius
            ax.add_patch(Circle(xy=(trimcat["GAL_XC"][0]-1, trimcat["GAL_YC"][0]-1),
                                radius=trimcat["GAL_REFF_XY"][0],
                                facecolor="none",edgecolor="lawngreen",lw=2,ls="--"))
            # # Add a ellipse showing the morph fit
            # ax.add_patch(Ellipse(xy=(trimcat["GAL_XC"][0]-1, trimcat["GAL_YC"][0]-1),
            #                      width=2*trimcat["GAL_SMA_XY"][0], height=2*trimcat["GAL_SMB_XY"][0],
            #                      angle=trimcat["GAL_THETA"][0],
            #                      edgecolor='lawngreen',facecolor='none',lw=1,ls="--"))

    for ax in [ax1,ax2,ax3,ax4]+axes:
        ax.set_xlim(segimg.shape[0]*0.05,segimg.shape[0]*0.95)
        ax.set_ylim(segimg.shape[1]*0.05,segimg.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    img = mpimg.imread("S82/jpegs/objID{0[survey_id]:020d}.jpg".format(entry))
    ax1.imshow(img)
    ax1.scatter(img.shape[0]/2,img.shape[1]/2,c="w",marker="+",lw=0.8,s=100,alpha=0.8)
    ax1.set_xlim(1, img.shape[0]-1)
    ax1.set_ylim(img.shape[1]-1, 1)

    img = mpimg.imread("DR14/jpegs/objID{0[survey_id]:020d}_raw.jpg".format(entry))
    angSize = img.shape[0] * JPEG_pixscale
    ax2.imshow(img)

    transform = lambda r,d: transformImgToWCS(r,d,entry["RA"],entry["DEC"],size=angSize,scale=JPEG_pixscale)
    if trimcat is not None:
        ax2.scatter(*transform(trimcat["GAL_RA"],trimcat["GAL_DEC"]),c="w",marker="+",s=250,lw=3,alpha=0.8)
        if markPts:
            ax2.scatter(*transform(fullcat["X_WORLD"],fullcat["Y_WORLD"]),s=50,color="tab:red",marker="x",lw=2)
            ax2.scatter(*transform(trimcat["RA"],trimcat["DEC"]),s=50,color="lawngreen",marker="x",lw=2)

    ax2.set_xlim(segimg.shape[0]*FITS_pixscale/JPEG_pixscale*0.05,
                 segimg.shape[0]*FITS_pixscale/JPEG_pixscale*0.95)
    ax2.set_ylim(segimg.shape[1]*FITS_pixscale/JPEG_pixscale*0.95,
                 segimg.shape[1]*FITS_pixscale/JPEG_pixscale*0.05)

    if savefig:
        fig.savefig("plots/summary_clumps/objID{:020d}-curate.png".format(entry["survey_id"]))
        plt.close(fig)
    else:
        return fig,[ax1,ax2,ax3,ax4],transform

def plotClumpsPhotometry(entry,destdir,photcat=None,savefig=True):

    fullcat = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-cat.fits".format(entry, destdir))
    if photcat is None:
        try:
            photcat = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-phot.fits".format(entry, destdir))
        except OSError:
            photcat = None
    elif len(photcat)==0:
        photcat = None

    fig, axes = plt.subplots(2,7,figsize=(21.25,6.5),dpi=75)
    fig.subplots_adjust(left=0.02/7,right=1-0.02/7,top=0.94,bottom=0.02/2,wspace=0,hspace=0)
    fig.suptitle("ID {:d} ({:s})".format(entry["survey_id"],destdir),fontsize=22,fontweight=600)

    segimg = fitsio.getdata("photom/{1:s}/objID{0[survey_id]:020d}-seg.fits".format(entry,destdir))
    segimg = reprocessSegMap(segimg)
    axes[0,1].imshow(segimg,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segimg),origin="lower")
    axes[0,1].text(0.05,0.95,"SEGM",ha="left",va="top",color="lime",fontsize=20,fontweight=800,transform=axes[0,1].transAxes)

    segimg = fitsio.getdata("photom/smooth/objID{0[survey_id]:020d}-seg.smooth.fits".format(entry))
    segimg = reprocessSegMap(segimg)
    axes[1,1].imshow(segimg,cmap=plt.cm.CMRmap,vmin=1,vmax=np.max(segimg),origin="lower")
    axes[1,1].text(0.05,0.95,"SEGM (smooth)",ha="left",va="top",color="lime",fontsize=20,fontweight=800,transform=axes[1,1].transAxes)

    if photcat is not None:
        apersize = getClumpApertureSize(entry=entry)
        apertures = getClumpApertures(photcat,entry=entry)
        clump_mask = getClumpMask(photcat,imshape=segimg.shape,radius=0.5*apersize["mask"])

    for j,filt in enumerate(sdss_filters):

        imgname = "S82/fits/objID{0[survey_id]:020d}-{1:s}.fits".format(entry,filt)
        img,img_hdr = fitsio.getdata(imgname,header=True)

        vmin,vmax = getVminVmax(img)
        axes[0,j+2].text(0.05,0.95,filt,ha="left",va="top",color="purple",fontsize=24,fontweight=800,transform=axes[0,j+2].transAxes)
        axes[0,j+2].imshow(img,origin="lower",cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)

        if photcat is not None:
            mask_img = img.copy()
            mask_img[clump_mask] = -99.0
            axes[1,j+2].imshow(mask_img,origin="lower",cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)
            apertures["annl"].plot(axes=axes[1,j+2],color="blue",lw=0.5)

    _axes = np.append(axes[0,1:],axes[1,1])
    if photcat is None:
        for ax in _axes:
            ax.scatter(fullcat["X_IMAGE"]-1,fullcat["Y_IMAGE"]-1,s=30,color="r",marker="x",lw=1)
    else:
        for ax in np.append(_axes,axes[1,2:]):
            # Mark all rejected "clumps"
            rejcat = fullcat[~np.in1d(fullcat["NUMBER"],photcat["ID"])]
            ax.scatter(rejcat["X_IMAGE"]-1,rejcat["Y_IMAGE"]-1,s=30,color="r",marker="x",lw=1)
            # Mark all clumps
            ax.scatter(photcat["X"]-1,photcat["Y"]-1,s=30,color="lawngreen",marker="x",lw=1)

            # Mark the computed CoM for the diffuse light
            ax.scatter(photcat["GAL_XC"][0]-1,photcat["GAL_YC"][0]-1,s=250,color="lawngreen",marker="+",lw=3)
            # Add a circle showing the petro radius
            ax.add_patch(Circle(xy=(photcat["GAL_XC"][0]-1, photcat["GAL_YC"][0]-1),
                                radius=photcat["GAL_REFF_XY"][0],
                                facecolor="none",edgecolor="lawngreen",lw=1,ls="--"))
            # Add a ellipse showing the morph fit
            ax.add_patch(Ellipse(xy=(photcat["GAL_XC"][0]-1, photcat["GAL_YC"][0]-1),
                                 width=2*photcat["GAL_SMA_XY"][0], height=2*photcat["GAL_SMB_XY"][0],
                                 angle=photcat["GAL_THETA"][0],
                                 edgecolor='lawngreen',facecolor='none',lw=1,ls="--"))


    for ax in axes.flatten():
        ax.set_xlim(segimg.shape[0]*0.05,segimg.shape[0]*0.95)
        ax.set_ylim(segimg.shape[1]*0.05,segimg.shape[1]*0.95)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    img = mpimg.imread("S82/jpegs/objID{0[survey_id]:020d}.jpg".format(entry))
    axes[0,0].imshow(img,origin="lower")
    axes[0,0].scatter(img.shape[0]/2,img.shape[1]/2,c="w",marker="+",lw=0.8,s=100,alpha=0.8)
    axes[0,0].set_xlim(1, img.shape[0]-1)
    axes[0,0].set_ylim(img.shape[1]-1, 1)

    img = mpimg.imread("DR14/jpegs/objID{0[survey_id]:020d}.jpg".format(entry))
    ang_size = img.shape[0] * JPEG_pixscale
    axes[1,0].imshow(img, origin="lower")
    axes[1,0].scatter(*transformImgToWCS(entry["RA"],entry["DEC"],entry["RA"],entry["DEC"],size=ang_size,scale=JPEG_pixscale),
                        c="w",marker="+",s=100,lw=0.8,alpha=0.8)
    axes[1,0].set_xlim(segimg.shape[0]*FITS_pixscale/JPEG_pixscale*0.05,
                       segimg.shape[0]*FITS_pixscale/JPEG_pixscale*0.95)
    axes[1,0].set_ylim(segimg.shape[1]*FITS_pixscale/JPEG_pixscale*0.95,
                       segimg.shape[1]*FITS_pixscale/JPEG_pixscale*0.05)

    if savefig:
        fig.savefig("plots/summary_clumps/objID{:020d}-photom.png".format(entry["survey_id"]))
        plt.close(fig)
    else:
        return fig, axes


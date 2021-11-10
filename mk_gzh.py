from useful import *

def getClumpyCondition(data):

    cond = (data["t01_smooth_or_features_a02_features_or_disk_weighted_fraction"] >= 0.5) & \
           (data["t12_clumpy_a01_yes_weighted_fraction"] >= 0.5) & \
           (data["t12_clumpy_total_count"] >= 20)
    return cond

def getNewDtype(dtype1,dtype2):
    new_dtype = []
    for x in dtype1+dtype2:
        if x[0] not in [_[0] for _ in new_dtype]:
            if x[0]=="survey_id": x = (x[0],'>i8')
            new_dtype.append(x)
    return new_dtype

def mergeGZHMetadata():

    for sampletype in ["single","coadd"]:

        cdata = fitsio.getdata("catalogs/gz_hubble_sdss_{:s}.fits".format(sampletype),1)
        mdata = fitsio.getdata("catalogs/metadata_sdss_{:s}.fits".format(sampletype),1)

        dtype = getNewDtype(cdata.dtype.descr,mdata.dtype.descr)
        merge = np.recarray(len(cdata),dtype=dtype)

        for x in merge.dtype.names:
            if x in mdata.dtype.names:
                merge[x] = mdata[x]
            if x in cdata.dtype.names:
                merge[x] = cdata[x]

        savename = "catalogs/gz_hubble_sdss_{:s}.merged.fits".format(sampletype)
        fitsio.writeto(savename, merge, overwrite=True)

def matchSingleVsCoadd():

    catSingle = fitsio.getdata("catalogs/gz_hubble_sdss_single.merged.fits")
    catCoadd  = fitsio.getdata("catalogs/gz_hubble_sdss_coadd.merged.fits")

    m1,m2,dist = matchRADec(catSingle["RA"],catSingle["DEC"],catCoadd["RA"],catCoadd["DEC"],crit=10,maxmatch=0)

    catCoadd  = rfn.append_fields( catCoadd[m2],names="distance",data=dist*3600,dtypes=float,usemask=False,asrecarray=True)
    catSingle = rfn.append_fields(catSingle[m1],names="distance",data=dist*3600,dtypes=float,usemask=False,asrecarray=True)

    hduList = fitsio.HDUList()
    hduList.append(fitsio.BinTableHDU(catCoadd,name="coadd"))
    hduList.append(fitsio.BinTableHDU(catSingle,name="single"))
    hduList.writeto("catalogs/gz_hubble_sdss.matched.fits",overwrite=True)

def deDuplicate(sample):

    m1,m2,dist = matchRADec(sample["RA"],sample["DEC"],sample["RA"],sample["DEC"],crit=20,maxmatch=0)
    cond = (m1!=m2)
    m1,m2 = m1[cond],m2[cond]

    magM1 = sample["PETROMAG_R"][m1]
    magM2 = sample["PETROMAG_R"][m2]
    cullID = np.concatenate([m1[magM1>magM2],m2[magM2>magM1]])
    sample = np.delete(sample,cullID)
    return sample

def customCoaddEntries(sample):

    ### Both of these are recovered from the single epoch sample
    removeID  = [8647474690891186220,8647474691387491512]

    for remove in removeID:
        idx = np.where(sample["survey_id"]==remove)[0][0]
        sample = np.delete(sample,idx)

    return sample

def mergeSingleCoadd():

    catSingle = fitsio.getdata("catalogs/gz_hubble_sdss_single.merged.fits")
    catSingle = catSingle[getClumpyCondition(catSingle)]
    catSingle = rfn.append_fields(catSingle,names="clumpy_selection",data=["single"]*len(catSingle),
                                    dtypes="U6",usemask=False,asrecarray=True)
    catSingle = deDuplicate(catSingle)

    catCoadd = fitsio.getdata("catalogs/gz_hubble_sdss_coadd.merged.fits")
    catCoadd = catCoadd[getClumpyCondition(catCoadd)]
    catCoadd = rfn.append_fields(catCoadd,names="clumpy_selection",data=["coadd"]*len(catCoadd),
                                    dtypes="U6",usemask=False,asrecarray=True)
    catCoadd = deDuplicate(catCoadd)
    catCoadd = customCoaddEntries(catCoadd)

    catSingleMatched = fitsio.getdata("catalogs/gz_hubble_sdss.matched.fits",extname="single")
    catCoaddMatched  = fitsio.getdata("catalogs/gz_hubble_sdss.matched.fits",extname="coadd")

    dtype = [x for x in catCoadd.dtype.descr if x[0].lower()[0]!='t'] + [("survey_id_single",int)]
    catClumpy = np.recarray(len(catSingle)+len(catCoadd),dtype=dtype)
    for x in catClumpy.dtype.names: catClumpy[x] = -99
    for x in catClumpy.dtype.names[:-1]: catClumpy[x][:len(catCoadd)] = catCoadd[x]

    condClumpySingleAndCoadd,condClumpySingleNotCoadd,condClumpySingleNotMatched = np.zeros((3,len(catSingle)),dtype=bool)
    for i,entry in enumerate(catSingle):
        idx = np.where(catSingleMatched["survey_id"]==entry["survey_id"])[0]
        if len(idx)>0:
            entryCoaddMatched = catCoaddMatched[idx]
            condCoaddMatched = np.in1d(entryCoaddMatched["survey_id"],catCoadd["survey_id"])
            if sum(condCoaddMatched)>0:
                condClumpySingleAndCoadd[i] = True
            else:
                condClumpySingleNotCoadd[i] = True
                idx = np.argmin(entryCoaddMatched["distance"]) if len(entryCoaddMatched)>1 else 0
                for x in catClumpy.dtype.names[:-2]: catClumpy[x][len(catCoadd)+i] = entryCoaddMatched[x][idx]
                catClumpy["clumpy_selection"][len(catCoadd)+i] = "single"
                catClumpy["survey_id_single"][len(catCoadd)+i] = entry["survey_id"]
        else:
            condClumpySingleNotMatched[i] = True

    condClumpyCoaddNotMatched = np.zeros(len(catCoadd),dtype=bool)
    for i,entry in enumerate(catCoadd):
        idx = np.where(catCoaddMatched["survey_id"]==entry["survey_id"])[0]
        if len(idx)>0:
            entryCoaddMatched = catCoaddMatched[idx]
            entrySingleMatched = catSingleMatched[idx]
            idx = np.argmin(entryCoaddMatched["distance"]) if len(idx)>1 else 0
            catClumpy["survey_id_single"][i] = entrySingleMatched["survey_id"][idx]
        else:
            condClumpyCoaddNotMatched[i] = True

    catClumpy = catClumpy[catClumpy["survey_id"]!=-99]

    print("Clumpy in Single-only: {:d}".format(len(catSingle)))
    print("Clumpy in  Coadd-only: {:d}".format(len(catCoadd)))
    print("Clumpy in Single and Coadd: {:d}".format(sum(condClumpySingleAndCoadd)))
    print("Clumpy in Single but not in Coadd: {:d}".format(sum(condClumpySingleNotCoadd)))
    print("Clumpy in Single and no  Coadd match: {:d}".format(sum(condClumpySingleNotMatched)))
    print("Clumpy in Coadd  and no Single match: {:d}".format(sum(condClumpyCoaddNotMatched)))
    print("Final combined Clumpy sample: {:d}".format(len(catClumpy)))

    fitsio.writeto("samples/gzh_sdss_clumpy.fits",catClumpy,overwrite=True)

def plotVoteFracComparison(clumpyCond=False):

    def plot2DHist(xdata,ydata,binsx,binsy,axis,cmap,lognorm=False,hist1d=False):

        bincx = 0.5*(binsx[1:]+binsx[:-1])
        bincy = 0.5*(binsy[1:]+binsy[:-1])
        hist2d = np.histogram2d(xdata,ydata,bins=[binsx,binsy])[0]
        hist2d = np.ma.masked_array(hist2d,mask=hist2d<=0)

        from matplotlib.colors import Normalize,LogNorm
        norm = LogNorm(vmin=1,vmax=np.ma.max(hist2d)) if lognorm else Normalize(vmin=1,vmax=np.ma.max(hist2d))
        im = axis.pcolormesh(binsx,binsy,hist2d.T,cmap=cmap,norm=norm)

        if hist1d:
            hist = np.histogram(xdata,bins=binsx)[0]
            axis.plot(bincx,hist/max(hist)*0.3,color='gray',lw=2.5,alpha=0.8,drawstyle="steps-mid")
            hist = np.histogram(ydata,bins=binsy)[0]
            axis.plot(hist/max(hist)*0.3,bincy,color='gray',lw=2.5,alpha=0.8,drawstyle="steps-mid")

        divider = make_axes_locatable(axis)
        cbaxes = divider.append_axes("right", size="4%", pad=0.1)
        cbax = fig.colorbar(mappable=im, cax=cbaxes, orientation="vertical")
        cbax.ax.tick_params(labelsize=14)

    catCoaddMatched = fitsio.getdata("catalogs/gz_hubble_sdss.matched.fits",extname="coadd")
    catSingleMatched = fitsio.getdata("catalogs/gz_hubble_sdss.matched.fits",extname="single")
    condCoaddMatched = getClumpyCondition(catCoaddMatched)
    condSingleMatched = getClumpyCondition(catSingleMatched)

    fig,axes = plt.subplots(2,3,figsize=(17,10),dpi=75,tight_layout=True)

    bins = np.arange(-1,2,0.025)
    plot2DHist(catCoaddMatched["t01_smooth_or_features_a02_features_or_disk_weighted_fraction"],
               catSingleMatched["t01_smooth_or_features_a02_features_or_disk_weighted_fraction"],
                binsx=bins,binsy=bins,axis=axes[0,0],cmap=plt.cm.inferno,lognorm=True)
    plot2DHist(catCoaddMatched["t12_clumpy_a01_yes_weighted_fraction"],
               catSingleMatched["t12_clumpy_a01_yes_weighted_fraction"],
                binsx=bins,binsy=bins,axis=axes[1,0],cmap=plt.cm.inferno,lognorm=True)
    axes[0,0].set_title("All GZ:H",fontsize=18,fontweight=600)

    for i,(cond,condtype) in enumerate(zip([condSingleMatched,condCoaddMatched],["Single","Coadd"])):

        bins = np.arange(-1,2,0.05)
        plot2DHist(catCoaddMatched["t01_smooth_or_features_a02_features_or_disk_weighted_fraction"][cond],
                   catSingleMatched["t01_smooth_or_features_a02_features_or_disk_weighted_fraction"][cond],
                    binsx=bins,binsy=bins,axis=axes[0,1+i],cmap=plt.cm.inferno,hist1d=True)
        plot2DHist(catCoaddMatched["t12_clumpy_a01_yes_weighted_fraction"][cond],
                   catSingleMatched["t12_clumpy_a01_yes_weighted_fraction"][cond],
                    binsx=bins,binsy=bins,axis=axes[1,1+i],cmap=plt.cm.inferno,hist1d=True)
        axes[0,1+i].set_title("Clumpy condition on {:s}".format(condtype),fontsize=18,fontweight=600)

    for i in range(3):
        axes[0,i].set_xlabel("Coadd $f_{Featured}$",fontsize=16)
        axes[0,i].set_ylabel("Single $f_{Featured}$",fontsize=16)
        axes[1,i].set_xlabel("Coadd $f_{Clumpy}$",fontsize=16)
        axes[1,i].set_ylabel("Single $f_{Clumpy}$",fontsize=16)

    for ax in axes.flatten():
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.plot([0,1],[0,1],c='w',lw=1.2)
        ax.set_aspect(1)
        [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig("plots/sample/compare_clumpyVoteFracs.png")

def plotStampComparison():

    def plotTwinStamps(idx,gs):

        if catSingleMatched["t12_clumpy_a01_yes_weighted_fraction"][idx] < \
           catCoaddMatched["t12_clumpy_a01_yes_weighted_fraction"][idx]:
            colorSingle,colorCoadd = "red","lawngreen"
        else:
            colorSingle,colorCoadd = "lawngreen","red"

        gssub = gs.subgridspec(1,2,hspace=0,wspace=0)
        ax1 = fig.add_subplot(gssub[0])
        img = mpimg.imread("/data/extragal/willett/gzh/jpg/{:d}.jpg".format(catSingleMatched["survey_id"][idx]))
        ax1.imshow(img)
        ax1.text(0.95,0.95,catSingleMatched["t12_clumpy_a01_yes_weighted_fraction"][idx],
                    color=colorSingle,fontsize=10,fontweight=600,ha="right",va="top",transform=ax1.transAxes)

        ax2 = fig.add_subplot(gssub[1])
        img = mpimg.imread("/data/extragal/willett/gzh/jpg/{:d}.jpg".format(catCoaddMatched["survey_id"][idx]))
        img = np.rot90(img,1)
        ax2.imshow(img)
        ax2.text(0.05,0.95,catCoaddMatched["t12_clumpy_a01_yes_weighted_fraction"][idx],
                    color=colorCoadd,fontsize=10,fontweight=600,ha="left",va="top",transform=ax2.transAxes)

        for ax in [ax1,ax2]:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        sample = fitsio.getdata("../process/samples/final_clumpy_sample.lowz.fits")
        if catSingleMatched["survey_id"][idx] in sample["survey_id"] or \
           catCoaddMatched["survey_id"][idx] in sample["survey_id"]:
            ax1.spines["left"].set_color("tab:red")
            ax1.spines["left"].set_linewidth(2)
            ax2.spines["right"].set_color("tab:red")
            ax2.spines["right"].set_linewidth(2)
            for ax in [ax1,ax2]:
                ax.spines["bottom"].set_color("tab:red")
                ax.spines["bottom"].set_linewidth(2)
                ax.spines["top"].set_color("tab:red")
                ax.spines["top"].set_linewidth(2)

    catCoaddMatched = fitsio.getdata("catalogs/gz_hubble_sdss.matched.fits",extname="coadd")
    catSingleMatched = fitsio.getdata("catalogs/gz_hubble_sdss.matched.fits",extname="single")

    condClumpy = getClumpyCondition(catCoaddMatched) | getClumpyCondition(catSingleMatched)
    catCoaddMatched = catCoaddMatched[condClumpy]
    catSingleMatched = catSingleMatched[condClumpy]

    condCoaddOnly = (catCoaddMatched["t12_clumpy_a01_yes_weighted_fraction"] > 0.5) & \
                    (catSingleMatched["t12_clumpy_a01_yes_weighted_fraction"] < 0.5)
    condSingleOnly = (catCoaddMatched["t12_clumpy_a01_yes_weighted_fraction"] < 0.5) & \
                     (catSingleMatched["t12_clumpy_a01_yes_weighted_fraction"] > 0.5)

    fig = plt.figure(figsize=(10*1.5,10*1.5),dpi=150)
    gsgrid = fig.add_gridspec(10,5)
    fig.subplots_adjust(left=0.01,right=0.99,bottom=0.01,top=0.99,wspace=0.02,hspace=0.02)
    for gs,idx in zip(gsgrid,np.where(condCoaddOnly)[0]): plotTwinStamps(idx,gs)
    fig.savefig("plots/sample/stamps_clumpyCoaddOnly.png")

    fig = plt.figure(figsize=(16*1.5,10*1.5),dpi=150)
    gsgrid = fig.add_gridspec(10,8)
    fig.subplots_adjust(left=0.01*8/10,right=1-0.01*8/10,bottom=0.01,top=0.99,wspace=0.02*8/10,hspace=0.02)
    for gs,idx in zip(gsgrid,np.where(condSingleOnly)[0]): plotTwinStamps(idx,gs)
    fig.savefig("plots/sample/stamps_clumpySingleOnly.png")

def plotClumpyStamps():

    def plotStamp(entry,gs):

        ax = fig.add_subplot(gs)
        img = mpimg.imread("/data/extragal/willett/gzh/jpg/{:d}.jpg".format(entry["survey_id"]))
        ax.imshow(img)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    catClumpy = fitsio.getdata("samples/gzh_sdss_clumpy.fits")

    fig = plt.figure(figsize=(22*1.5,20*1.5),dpi=150)
    gsgrid = fig.add_gridspec(20,22)
    fig.subplots_adjust(left=0.01*20/22,right=1-0.01*20/22,bottom=0.01,top=0.99,wspace=0.02*20/22,hspace=0.02)
    for gs,entry in zip(gsgrid,catClumpy): plotStamp(entry,gs)
    fig.savefig("plots/sample/stamps_clumpyGZH.png")

if __name__ == '__main__':

    # mergeGZHMetadata()
    # matchSingleVsCoadd()
    mergeSingleCoadd()

    # plotVoteFracComparison()
    # plotStampComparison()
    # plotClumpyStamps()

    # plt.show()


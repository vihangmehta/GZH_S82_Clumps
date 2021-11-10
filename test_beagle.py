from useful import *

def plot_median(axis,xdata,ydata,c,nbin,s=0,marker='o',label=None,xerr=True,yerr=True,alpha=0.9,zorder=10,fill=True,logify=True,xxdata=None):

    if xxdata is None: xxdata = xdata
    tmpx = np.log10(xxdata) if logify else xxdata
    tmp  = np.array_split(np.sort(tmpx),len(xxdata)//nbin)
    bins = np.array([_[0] for _ in tmp] + [tmp[-1][-1]])
    if logify: bins = 10**bins
    binc = 0.5*(bins[1:] + bins[:-1])

    idig = np.digitize(xdata, bins) - 1

    l68,med,u68 = np.zeros((3,len(binc)))
    med = med * np.NaN
    for i in range(len(binc)):
        if sum(idig==i) >= 5:
            l68[i],med[i],u68[i] = np.nanpercentile(ydata[idig==i],[16,50,84])

    print(med)
    yerr = [med - l68, u68 - med]
    axis.errorbar(binc,med,#yerr=yerr,
                    color=c, marker=marker, markersize=s, mew=1, lw=5,
                    ecolor=c, elinewidth=1, capsize=3, alpha=alpha, zorder=zorder, label=label)
    axis.plot(binc,l68,color=c,lw=1.5,ls='--',alpha=alpha)
    axis.plot(binc,u68,color=c,lw=1.5,ls='--',alpha=alpha)
    if fill: axis.fill_between(binc,l68,u68,color=c,lw=0,alpha=0.1)

def compare_beagle_output(catalog1,catalog2,label1,label2):

    catalog1 = fitsio.getdata(catalog1)
    catalog2 = fitsio.getdata(catalog2)

    fig1,axes1 = plt.subplots(2,3,figsize=(12,8),dpi=75,tight_layout=True)
    fig2,axes2 = plt.subplots(2,3,figsize=(12,8),dpi=75,tight_layout=True)
    axes1,axes2 = axes1.flatten(),axes2.flatten()

    plims = {"logMst":[3.8,9.6],"logage":[5.8,10.2],"Z_star":[-1.55,-0.18],"logOH":[7.15,8.55],"logU":[-3.65,-2.25],"AV":[-0.1,4.4]}
    elims = {"logMst":[-0.1,2.7],"logage":[-0.1,3.6],"Z_star":[-0.05,1.1],"logOH":[-0.05,1.1],"logU":[-0.05,1.3],"AV":[-0.1,1.6]}

    for i,par in enumerate(["logMst","logage","Z_star","logOH","logU","AV"]):

        axes1[i].scatter(catalog1[par],catalog2[par],color='k',lw=0,s=10,alpha=0.8)
        axes1[i].plot(plims[par],plims[par],c='k',lw=0.8,alpha=0.8)
        axes1[i].set_xlim(plims[par])
        axes1[i].set_ylim(plims[par])
        axes1[i].set_xlabel("Best-fit {:s} [{:s}]".format(par,label1),fontsize=16)
        axes1[i].set_ylabel("Best-fit {:s} [{:s}]".format(par,label2),fontsize=16)
        axes1[i].set_aspect(1)

        axes2[i].scatter(catalog1[par+"_u68"]+catalog1[par+"_l68"],
                         catalog2[par+"_u68"]+catalog2[par+"_l68"],color='k',lw=0,s=10,alpha=0.8)
        axes2[i].plot(elims[par],elims[par],c='k',lw=0.8,alpha=0.8)
        axes2[i].set_xlim(elims[par])
        axes2[i].set_ylim(elims[par])
        axes2[i].set_xlabel("{:s} error [{:s}]".format(par,label1),fontsize=16)
        axes2[i].set_ylabel("{:s} error [{:s}]".format(par,label2),fontsize=16)
        axes2[i].set_aspect(1)

def compare_age_gradient(fincat1,catalog1,label1,
                         fincat2,catalog2,label2,
                         fincat3,catalog3,label3):

    fincat1 = fitsio.getdata(fincat1)
    fincat2 = fitsio.getdata(fincat2)
    fincat3 = fitsio.getdata(fincat3)

    catalog1 = fitsio.getdata(catalog1)
    catalog2 = fitsio.getdata(catalog2)
    catalog3 = fitsio.getdata(catalog3)
    cond = (catalog3["ID"]!=-99)

    fig,ax = plt.subplots(1,1,figsize=(10,8),dpi=75,tight_layout=True)

    plot_median(xdata=fincat1["clump_distnorm"],ydata=catalog1["logage"],axis=ax,c="tab:red", nbin=40,label=label1)
    plot_median(xdata=fincat2["clump_distnorm"],ydata=catalog2["logage"],axis=ax,c="tab:blue", nbin=40,label=label2)
    plot_median(xdata=fincat3[      "distnorm"][cond],ydata=catalog3["logage"][cond],axis=ax,c="tab:green", nbin=40,label=label3)#,xxdata=fincat1["clump_distnorm"])

    ax.set_ylim(5.8,10.2)
    ax.set_ylabel("log age [yr]",fontsize=18)
    ax.set_xlim(1.5e-1,1e1)
    ax.set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=18)
    ax.set_xscale('log')
    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    ax.legend(fontsize=14)
    [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

def temp(fincat1,catalog1,label1,
         fincat2,catalog2,label2,
         fincat3,catalog3,label3):

    fincat1 = fitsio.getdata(fincat1)
    fincat2 = fitsio.getdata(fincat2)
    fincat3 = fitsio.getdata(fincat3)

    catalog1 = fitsio.getdata(catalog1)
    catalog2 = fitsio.getdata(catalog2)
    catalog3 = fitsio.getdata(catalog3)

    fig,axes = plt.subplots(2,2,figsize=(18,12),dpi=75,sharex=True,tight_layout=True)
    axes = axes.flatten()

    axes[0].scatter(fincat1["clump_distnorm"],catalog1["logage"],c='tab:red',s=10,lw=0,alpha=0.8)
    axes[1].scatter(fincat2["clump_distnorm"],catalog2["logage"],c='tab:blue',s=10,lw=0,alpha=0.8)
    axes[2].scatter(fincat3[      "distnorm"],catalog3["logage"],c='tab:green',s=10,lw=0,alpha=0.8)

    plot_median(xdata=fincat1["clump_distnorm"],ydata=catalog1["logage"],axis=axes[0],c="tab:red", nbin=40,label=label1)
    plot_median(xdata=fincat2["clump_distnorm"],ydata=catalog2["logage"],axis=axes[1],c="tab:blue", nbin=40,label=label2)
    plot_median(xdata=fincat3[      "distnorm"],ydata=catalog3["logage"],axis=axes[2],c="tab:green", nbin=40,label=label3)#,xxdata=fincat1["clump_distnorm"])

    fincat4  = fitsio.getdata("photom/diffgal/diffgal_photom.fits")
    catalog4 = fitsio.getdata("beagle/data/clumps_beagle_output.diffgal.fits")

    gal_id,iuniq = np.unique(fincat4["GAL_ID"],return_index=True)
    idx = np.argsort(iuniq)
    for gid,color in zip(gal_id[idx],plt.cm.Set1(np.linspace(0,1,9))):
        cond = (fincat4["GAL_ID"]==gid)
        _fincat4,_catalog4 = fincat4[cond],catalog4[cond]
        _fincat4["DISTNORM"] = np.round(_fincat4["DISTNORM"],2)
        dist = np.unique(_fincat4["DISTNORM"])
        mean = np.array([np.median(_catalog4["logage"][_fincat4["DISTNORM"]==x]) for x in dist])
        axes[3].plot(dist,mean,marker='o',markersize=8,mew=0,lw=2,alpha=0.8,label="ID#{:d}".format(gid))

        # isort = np.argsort(fincat3["DISTNORM"][cond])
        # axes[3].plot(fincat3["DISTNORM"][cond][isort],catalog3["logage"][cond][isort],lw=2,alpha=0.8)
        # plot_median(xdata=fincat3["DISTNORM"][cond][isort],ydata=catalog3["logage"][cond][isort],axis=axes[3],c=color,nbin=8)

    axes[1].set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=18)
    axes[3].set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=18)
    axes[3].legend(fontsize=14,loc="best")

    for ax in axes.flatten():
        ax.set_ylim(5.8,10.2)
        ax.set_ylabel("log age [yr]",fontsize=18)
        ax.set_xscale('log')
        ax.set_xlim(1.5e-1,1e1)
        xticks = [0.2,0.3,0.4,0.5,0.6,0.8,1,2,3,4,5,6,8]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        # ax.legend(fontsize=14)
        [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

if __name__ == '__main__':

    # catalog1,label1 = "beagle/data/clumps_beagle_output.galaxy.fits","Galaxy [AUTO]"
    # catalog2,label2 = "beagle/data/clumps_beagle_output.galiso.fits","Galaxy [ISO]"
    # compare_beagle_output(catalog1=catalog1,label1=label1,catalog2=catalog2,label2=label2)

    fincat1 = "samples/final_clump_catalog.fits"
    fincat2 = "samples/final_clump_catalog.fits"
    fincat3 = "photom/diffgal/diffgal_photom.old.fits"

    catalog1,label1 = "beagle/data/clumps_beagle_output.const.fits", "Clumps"
    # catalog2,label2 = "beagle/data/clumps_beagle_output.errfix.fits","Clumps [errfix]"
    catalog2,label2 = "beagle/data/clumps_beagle_output.galsub.fits","Underlying galaxy"
    catalog3,label3 = "beagle/data/temp/clumps_beagle_output.diffgal.fits","Pure galaxy"

    # compare_age_gradient(fincat1=fincat1,catalog1=catalog1,label1=label1,
    #                      fincat2=fincat2,catalog2=catalog2,label2=label2,
    #                      fincat3=fincat3,catalog3=catalog3,label3=label3)

    temp(fincat1=fincat1,catalog1=catalog1,label1=label1,
                         fincat2=fincat2,catalog2=catalog2,label2=label2,
                         fincat3=fincat3,catalog3=catalog3,label3=label3)


    plt.show()

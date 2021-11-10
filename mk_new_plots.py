from useful import *

from mk_pretty_plots import getBeagleErrors, plot_median

def getVminVmax(img,sigclip=False):

    size = img.shape[0]
    _img = img[int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0)),
               int(np.floor(size / 3.0)) : int(np.ceil(size * 2.0 / 3.0))]
    _img = np.ma.masked_array(_img, mask=~np.isfinite(_img))
    if sigclip:
        _img = _img[(_img<np.ma.median(_img)+3*np.ma.std(_img)) & \
                    (_img>np.ma.median(_img)-3*np.ma.std(_img))]
    vmin = np.ma.median(_img) - 2.0*np.ma.std(_img)
    vmax = np.ma.median(_img) + 5.0*np.ma.std(_img)
    return vmin, vmax

def plot_mass_redshift_dist(catalog,guo_cat):

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,14),dpi=300,gridspec_kw={"height_ratios":[2,3,3]},tight_layout=True)
    bins = np.arange(5,15,0.1)

    uniq_gals,iuniq = np.unique(catalog["gzh_id"],return_index=True)
    ax1.hist(catalog["gal_logMst"][iuniq],bins=bins,color='k',density=True,alpha=0.5,label="S82 clumpy galaxies (z<0.06)")
    ax2.scatter(catalog["gal_redshiftdr14"][iuniq],catalog["gal_logMst"][iuniq],color='k',s=50,lw=0,alpha=0.8)

    gzhcat = fitsio.getdata("samples/final_clumpy_sample.fits")
    idx = [np.where(_["gzh_id"]==gzhcat["survey_id"])[0][0] for _ in catalog[iuniq]]
    ax3.scatter(gzhcat["PETROR50_R_KPC"][idx],catalog["gal_logMst"][iuniq],color='k',s=50,lw=0,alpha=0.8)

    uniq_gals,iuniq = np.unique(guo_cat["GalID"],return_index=True)
    ax1.hist(guo_cat["logM*"][iuniq],bins=bins,color='tab:green',lw=5,density=True,histtype="step",label="Guo+18 clumpy galaxies (0.5<z<3)")

    ax1.set_xlim(7.3,11.6)
    ax1.set_ylim(0,1.25)
    ax1.set_xlabel("Host galaxy stellar mass, log $M^\\star [M_\\odot]$",fontsize=24)
    ax1.set_ylabel("Norm. freq.",fontsize=24)

    leg = ax1.legend(loc=1,fontsize=18,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.6)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor())
        txt.set_fontweight(600)
        txt.set_alpha(1)
        hndl.set_visible(False)

    ax2.set_xlim(0,0.065)
    ax2.set_ylim(7.3,11.3)
    ax2.set_xlabel("Redshift, $z$",fontsize=24)
    ax2.set_ylabel("Host galaxy stellar mass,\nlog $M^\\star [M_\\odot]$",fontsize=24)

    ax3.set_xlim(0,7)
    ax3.set_ylim(7.3,11.3)
    ax3.set_xlabel("Petrosian radius, $r_{50}$ [kpc]",fontsize=24)
    ax3.set_ylabel("Host galaxy stellar mass,\nlog $M^\\star [M_\\odot]$",fontsize=24)

    for ax in [ax1,ax2,ax3]:
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"galaxy_dist2.png"))

def plot_mass_age_radius(catalog,logify=True):

    beagle_galsub = fitsio.getdata("beagle/data/clumps_beagle_output.galsub.fits")

    fig = plt.figure(figsize=(11,6),dpi=300)
    fig.subplots_adjust(left=0.09,bottom=0.12,top=0.98,right=0.99)

    ogs = fig.add_gridspec(3,2,width_ratios=[4,1],wspace=0.005,hspace=0.05)
    ax  = fig.add_subplot(ogs[:,0])
    dax = [fig.add_subplot(ogs[i,1]) for i in range(3)]

    ax.set_ylim(5.5,10.5)
    ax.set_ylabel("log age [yr]",fontsize=24)
    ax.set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=24)

    if logify:
        ax.set_xlim(1.5e-1,1e1)
        ax.set_xscale('log')
    else:
        ax.set_xlim(0,5.5)
    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    ### Ages vs radius
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_distnorm",y="clump_logage")
    ax.errorbar(catalog["clump_distnorm"],catalog["clump_logage"],yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=catalog["clump_logage"],axis=ax,c="k",nbin=50,fill=True,logify=logify,label="Clumps")

    ### Plotting the mass and ages of the diffuse galaxy light
    xerr,yerr = getBeagleErrors(catalog=beagle_galsub,x="logMst",y="logMst")
    ax.errorbar(catalog["clump_distnorm"],beagle_galsub["logage"],yerr=yerr,
                    color="tab:red",marker='o',markersize=4,mew=0,lw=0,ecolor="tab:red",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=beagle_galsub["logage"],axis=ax,c="tab:red",nbin=50,fill=True,logify=logify,label="Underlying galaxy")

    ### Plotting the individual galaxies
    fincat  = fitsio.getdata("photom/diffgal/diffgal_photom.fits")
    catalog = fitsio.getdata("beagle/data/clumps_beagle_output.diffgal.fits")

    galIDs = [8647475119817098026, 8647474691403874778, 8647475120909451490]
    colors = ["tab:blue","tab:green","tab:purple"]
    for _dax,gid,color in zip(dax,galIDs,colors):
        cond = (fincat["GAL_ID"]==gid)
        _fincat,_catalog = fincat[cond],catalog[cond]
        _fincat["DISTNORM"] = np.round(_fincat["DISTNORM"],2)
        dist = np.unique(_fincat["DISTNORM"])
        dist = dist[dist>0.301]
        l68,med,u68 =  np.array([np.percentile(_catalog["logage"][_fincat["DISTNORM"]==x],[16,50,84]) for x in dist]).T

        yerr = [med - l68, u68 - med]
        ax.errorbar(dist,med,yerr=yerr,
                        color=color, marker='o', markersize=0, mew=1, lw=3,
                        ecolor=color, elinewidth=1, capsize=3, alpha=0.9, zorder=10, label="ID#{:d}".format(gid))

        # img = mpimg.imread("DR14/jpegs/objID{:020d}_raw.jpg".format(gid))
        # _dax.imshow(img)

        img = fitsio.getdata("S82/fits/objID{:020d}-r.fits".format(gid))
        vmin,vmax = getVminVmax(img)
        _dax.imshow(img,cmap=plt.cm.Greys,vmin=vmin,vmax=vmax)

        _dax.text(0.02,0.98,gid,ha="left",va="top",color="k",fontsize=8.5,fontweight=800,transform=_dax.transAxes)
        _dax.set_xlim(img.shape[0]*0.25,img.shape[0]*0.75)
        _dax.set_ylim(_dax.get_xlim())
        _dax.xaxis.set_visible(False)
        _dax.yaxis.set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    leg = ax.legend(handles,labels,loc="best",fontsize=14,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.8)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_color())
        txt.set_fontweight(600)
        hndl.set_visible(False)

    savename = "clump_mass_age_radius.png"
    if not logify: savename = savename.replace(".png","_nonlog.png")
    fig.savefig(os.path.join(savedir,savename))

def plot_Av_metal_radius(catalog,logify=True):

    beagle_galsub = fitsio.getdata("beagle/data/clumps_beagle_output.galsub.fits")

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(9,8),dpi=75,sharex=True)
    fig.subplots_adjust(left=0.11,bottom=0.1,top=0.98,right=0.98,wspace=0.03,hspace=0.03)

    ### Mass vs radius
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_distnorm",y="clump_AV")
    ax1.errorbar(catalog["clump_distnorm"],catalog[ "clump_AV"],yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=catalog[ "clump_AV"],axis=ax1,c="k",nbin=50,fill=True,logify=logify,label="Clumps")

    ### Ages vs radius
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_distnorm",y="clump_Z_star")
    ax2.errorbar(catalog["clump_distnorm"],catalog["clump_Z_star"],yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=catalog["clump_Z_star"],axis=ax2,c="k",nbin=50,fill=True,logify=logify,label="Clumps")

    ### Plotting the mass and ages of the diffuse galaxy light
    # xerr,yerr = getBeagleErrors(catalog=beagle_galsub,x="AV",y="AV")
    # ax1.errorbar(catalog["clump_distnorm"],beagle_galsub["AV"],yerr=yerr,
    #                 color="tab:red",marker='o',markersize=5,mew=0,lw=0,ecolor="k",elinewidth=0.2,capsize=0,alpha=0.25,zorder=8)
    # plot_median(xdata=catalog["clump_distnorm"],ydata=beagle_galsub["AV"],axis=ax1,c="tab:red",nbin=50,fill=True,logify=logify,label="Underlying galaxy")

    # xerr,yerr = getBeagleErrors(catalog=beagle_galsub,x="Z_star",y="Z_star")
    # ax2.errorbar(catalog["clump_distnorm"],beagle_galsub["Z_star"],yerr=yerr,
    #                 color="tab:red",marker='o',markersize=5,mew=0,lw=0,ecolor="k",elinewidth=0.2,capsize=0,alpha=0.25,zorder=8)
    # plot_median(xdata=catalog["clump_distnorm"],ydata=beagle_galsub["Z_star"],axis=ax2,c="tab:red",nbin=50,fill=True,logify=logify,label="Underlying galaxy")

    ax1.set_ylim(-0.1,4.2)
    ax1.set_ylabel("A$_V$",fontsize=24)
    ax2.set_ylim(-2.4,0.5)
    ax2.set_ylabel("Metallicity",fontsize=24)
    ax2.set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=24)

    for ax in [ax1,ax2]:
        if logify:
            ax.set_xlim(1.5e-1,1e1)
            ax.set_xscale('log')
        else:
            ax.set_xlim(0,5.5)
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    # handles, labels = ax2.get_legend_handles_labels()
    # handles = [h[0] for h in handles]
    # leg = ax2.legend(handles,labels,loc="best",fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.8)
    # for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
    #     txt.set_color(hndl.get_color())
    #     txt.set_fontweight(600)
    #     hndl.set_visible(False)

    savename = "clump_Av_metal_radius.png"
    if not logify: savename = savename.replace(".png","_nonlog.png")
    fig.savefig(os.path.join(savedir,savename))

def plot_Av_metal_age(catalog):

    beagle_galsub = fitsio.getdata("beagle/data/clumps_beagle_output.galsub.fits")

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(9,8),dpi=75,sharex=True)
    fig.subplots_adjust(left=0.11,bottom=0.1,top=0.98,right=0.98,wspace=0.03,hspace=0.03)

    ### Mass vs radius
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_logage",y="clump_AV")
    ax1.errorbar(catalog["clump_logage"],catalog[ "clump_AV"],yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_logage"],ydata=catalog[ "clump_AV"],axis=ax1,c="k",nbin=50,fill=True,logify=False,label="Clumps")

    ### Ages vs radius
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_logage",y="clump_Z_star")
    ax2.errorbar(catalog["clump_logage"],catalog["clump_Z_star"],yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_logage"],ydata=catalog["clump_Z_star"],axis=ax2,c="k",nbin=50,fill=True,logify=False,label="Clumps")

    ### Plotting the mass and ages of the diffuse galaxy light
    # xerr,yerr = getBeagleErrors(catalog=beagle_galsub,x="logage",y="AV")
    # ax1.errorbar(beagle_galsub["logage"],beagle_galsub["AV"],yerr=yerr,
    #                 color="tab:red",marker='o',markersize=5,mew=0,lw=0,ecolor="k",elinewidth=0.2,capsize=0,alpha=0.25,zorder=8)
    # plot_median(xdata=beagle_galsub["logage"],ydata=beagle_galsub["AV"],axis=ax1,c="tab:red",nbin=50,fill=True,logify=False,label="Underlying galaxy")

    # xerr,yerr = getBeagleErrors(catalog=beagle_galsub,x="logage",y="Z_star")
    # ax2.errorbar(beagle_galsub["logage"],beagle_galsub["Z_star"],yerr=yerr,
    #                 color="tab:red",marker='o',markersize=5,mew=0,lw=0,ecolor="k",elinewidth=0.2,capsize=0,alpha=0.25,zorder=8)
    # plot_median(xdata=beagle_galsub["logage"],ydata=beagle_galsub["Z_star"],axis=ax2,c="tab:red",nbin=50,fill=True,logify=False,label="Underlying galaxy")

    ax1.set_ylim(-0.1,4.2)
    ax1.set_ylabel("A$_V$",fontsize=24)
    ax2.set_ylim(-2.4,0.5)
    ax2.set_ylabel("Metallicity",fontsize=24)
    ax2.set_xlabel("log Age [yr]",fontsize=24)

    for ax in [ax1,ax2]:
        ax.set_xlim(5.5,10.5)
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    # handles, labels = ax2.get_legend_handles_labels()
    # handles = [h[0] for h in handles]
    # leg = ax2.legend(handles,labels,loc="best",fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.8)
    # for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
    #     txt.set_color(hndl.get_color())
    #     txt.set_fontweight(600)
    #     hndl.set_visible(False)

    savename = "clump_Av_metal_age.png"
    fig.savefig(os.path.join(savedir,savename))

def plot_age_radius_kpc(catalog,logify=True):

    beagle_galsub = fitsio.getdata("beagle/data/clumps_beagle_output.galsub.fits")

    fig,ax = plt.subplots(1,1,figsize=(9,5),dpi=300,sharex=True)
    fig.subplots_adjust(left=0.09,bottom=0.15,top=0.98,right=0.98,wspace=0.03,hspace=0.03)

    ### Ages vs radius
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_distphys",y="clump_logage")
    ax.errorbar(catalog["clump_distphys"],catalog["clump_logage"],yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distphys"],ydata=catalog["clump_logage"],axis=ax,c="k",nbin=50,fill=True,logify=logify,label="Clumps")

    ax.set_ylim(5.5,10.5)
    ax.set_ylabel("log age [yr]",fontsize=24)
    ax.set_xlabel("Distance [kpc]",fontsize=24)

    if logify:
        ax.set_xlim(4e-1,3e1)
        ax.set_xscale('log')
        xticks = [0.4,0.6,0.8,1,2,3,4,5,6,8,10,20,30]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
    else:
        ax.set_xlim(0,17)
    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    # handles, labels = ax.get_legend_handles_labels()
    # handles = [h[0] for h in handles]
    # leg = ax.legend(handles,labels,loc="best",fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.8)
    # for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
    #     txt.set_color(hndl.get_color())
    #     txt.set_fontweight(600)
    #     hndl.set_visible(False)

    savename = "clump_age_radius_kpc.png"
    if not logify: savename = savename.replace(".png","_nonlog.png")
    fig.savefig(os.path.join(savedir,savename))

if __name__ == '__main__':

    savedir = "plots/new/"
    catalog = fitsio.getdata("samples/final_clump_catalog.fits")

    guo_cat = fitsio.getdata("catalogs/guo/Guo17_Clumps_Catalog.fits")
    guocond = (guo_cat["f_zphot"]!=1) & (guo_cat["VRFlag"]!=1) & (guo_cat["BMFlag"]!=1) & (guo_cat["EFlag"]!=1)
    guo_cat = guo_cat[guocond]
    guo_cat["logA"] = guo_cat["logA"] + 9

    # plot_mass_redshift_dist(catalog,guo_cat)
    # plot_mass_age_radius(catalog,logify=True)
    plot_Av_metal_radius(catalog,logify=True)
    plot_Av_metal_age(catalog)
    plot_age_radius_kpc(catalog,logify=True)

    # plt.show()

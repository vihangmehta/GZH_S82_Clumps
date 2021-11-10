from useful import *

from matplotlib import ticker
from scipy.ndimage import gaussian_filter

def getBeagleErrors(catalog,x,y):

    xerr = np.zeros((2,len(catalog)))
    yerr = np.zeros((2,len(catalog)))

    cond = (catalog[x]!=-99) & (catalog[y]!=-99)
    catalog = catalog[cond]

    try:
        xerr[:,cond] = [-catalog[x+"_l68"][cond],catalog[x+"_u68"][cond]]
    except KeyError:
        xerr = None

    try:
        yerr[:,cond] = [-catalog[y+"_l68"][cond],catalog[y+"_u68"][cond]]
    except KeyError:
        yerr = None

    return xerr,yerr

def plot_median(axis,xdata,ydata,c,nbin,s=0,marker='o',label=None,xerr=True,yerr=True,alpha=0.9,zorder=10,fill=True,logify=True,mean_err=False,mean_err_loc=[0,0]):

    tmpx = np.log10(xdata) if logify else xdata
    tmp  = np.array_split(np.sort(tmpx),len(xdata)//nbin)
    bins = np.array([_[0] for _ in tmp] + [tmp[-1][-1]])
    if logify: bins = 10**bins
    binc = 0.5*(bins[1:] + bins[:-1])

    idig = np.digitize(xdata, bins) - 1

    l68,med,u68 = np.zeros((3,len(binc)))
    med = med * np.NaN
    for i in range(len(binc)):
        if sum(idig==i) >= 5:
            l68[i],med[i],u68[i] = np.percentile(ydata[idig==i],[16,50,84])

    yerr = [med - l68, u68 - med]
    axis.errorbar(binc,med,#yerr=yerr,
                    color=c, marker=marker, markersize=s, mew=1, lw=5,
                    ecolor=c, elinewidth=1, capsize=3, alpha=alpha, zorder=zorder, label=label)
    axis.plot(binc,l68,color=c,lw=1.5,ls='--',alpha=alpha)
    axis.plot(binc,u68,color=c,lw=1.5,ls='--',alpha=alpha)
    if fill: axis.fill_between(binc,l68,u68,color=c,lw=0,alpha=0.1)
    if mean_err: axis.errorbar(*mean_err_loc,yerr=[[np.median(yerr[0])],[np.median(yerr[1])]],
                                color=c,marker='',ecolor=c,elinewidth=3,capsize=5,alpha=alpha,zorder=zorder)

def plot_pretty_stamps(catalog,jpg="DR14"):

    plotID = [8647474691387491507,8647475120900800722,8647475121426465287,8647475122534678723,8647474690858025275,
              8647474690337538126,8647474690867200065,8647474690891186212,8647474691419275289,8647475122536382735]
    plotID = np.array(plotID,dtype=int)

    clumpyVoteFrac, featureVoteFrac, pixScale = np.zeros((3,len(plotID)))
    catSingleMatched = fitsio.getdata("catalogs/gz_hubble_sdss.matched.fits",extname="single")
    catCoaddMatched  = fitsio.getdata("catalogs/gz_hubble_sdss.matched.fits",extname="coadd")
    for i,galID in enumerate(plotID):
        entry = catalog[catalog["gzh_id"]==galID][0]
        if entry["gzh_clumpy_selection"]=="single":
            idx = np.where(catSingleMatched["survey_id"]==entry["gzh_id_single"])[0][0]
            clumpyVoteFrac[i]  = catSingleMatched["t12_clumpy_a01_yes_weighted_fraction"][idx]
            featureVoteFrac[i] = catSingleMatched["t01_smooth_or_features_a02_features_or_disk_weighted_fraction"][idx]
        elif entry["gzh_clumpy_selection"]=="coadd":
            idx = np.where(catCoaddMatched["survey_id"]==entry["gzh_id"])[0][0]
            clumpyVoteFrac[i]  = catCoaddMatched["t12_clumpy_a01_yes_weighted_fraction"][idx]
            featureVoteFrac[i] = catCoaddMatched["t01_smooth_or_features_a02_features_or_disk_weighted_fraction"][idx]
        pixScale[i] = 0.02*catCoaddMatched["PETROR90_R"][idx]

    fig,axes = plt.subplots(2,5,figsize=(20,8),dpi=300)
    fig.subplots_adjust(left=0.005,right=0.995,bottom=0.01,top=0.99,wspace=0.01,hspace=0.01)

    for i,(objid,ax) in enumerate(zip(plotID,axes.flatten())):

        if   jpg== "S82": img = mpimg.imread("S82/jpegs/objID{:020d}.jpg".format(objid))
        elif jpg=="DR14": img = mpimg.imread("DR14/jpegs/objID{:020d}_raw.jpg".format(objid))
        img = np.flip(img,0)

        ax.imshow(img)
        ax.text(0.02,0.98,objid,ha="left",va="top",color="lime",fontsize=18,fontweight=800,transform=ax.transAxes)
        ax.text(0.02,0.02,"$f_{{featured}}={:.2f}$\n$f_{{clumpy}}={:.2f}$".format(featureVoteFrac[i],clumpyVoteFrac[i]),
                                ha="left",va="bottom",color="lime",fontsize=16,fontweight=800,transform=ax.transAxes)

        ax.set_xlim(img.shape[0]*0.15,img.shape[0]*0.85)
        if objid in plotID[[0,5,6,7]]:
            ax.set_xlim(img.shape[0]*0.25,img.shape[0]*0.75)

        ax.set_ylim(ax.get_xlim())
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        axlim = ax.get_xlim()
        dxlim = np.abs(np.diff(axlim))
        barx0,bary0,barh = axlim[1]-dxlim*0.08, axlim[0]+dxlim*0.08, dxlim*0.01

        if jpg=="S82":
            JPEG_pixscale = pixScale[i]
            ax.add_patch(Rectangle(xy=(barx0-5/JPEG_pixscale,bary0),width=5/JPEG_pixscale,height=barh,facecolor='w',edgecolor='none',lw=0,alpha=0.9))
            # ax.text(barx0-5/JPEG_pixscale/2, bary0+1.2*barh, "5\"", va="top", ha="center", fontsize=13, color="w")

        elif jpg=="DR14":
            JPEG_pixscale = 0.12
            ax.add_patch(Rectangle(xy=(barx0-5/JPEG_pixscale,bary0),width=5/JPEG_pixscale,height=barh,facecolor='w',edgecolor='none',lw=0,alpha=0.9))
            # ax.text(barx0-5/JPEG_pixscale/2, bary0+1.2*barh, "5\"", va="top", ha="center", fontsize=13, color="w")

            clumps = catalog[catalog["gzh_id"]==objid]
            for clump in clumps:
                pos = ((clump["clump_x"]-1)/JPEG_pixscale*FITS_pixscale,
                       (clump["clump_y"]-1)/JPEG_pixscale*FITS_pixscale)
                ax.add_patch(Circle(xy=pos,radius=clump["clump_psf_avg"]/JPEG_pixscale,lw=1,facecolor="none",edgecolor="lime"))

    fig.savefig(os.path.join(savedir,"clumpy_examples_%s.png"%jpg))

def plot_mass_redshift_dist(catalog,guo_cat):

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,10),dpi=300,gridspec_kw={"height_ratios":[2,3]},tight_layout=True)
    bins = np.arange(5,15,0.1)

    uniq_gals,iuniq = np.unique(catalog["gzh_id"],return_index=True)
    ax1.hist(catalog["gal_logMst"][iuniq],bins=bins,color='k',density=True,alpha=0.5,label="S82 clumpy galaxies (z<0.06)")
    ax2.scatter(catalog["gal_redshiftdr14"][iuniq],catalog["gal_logMst"][iuniq],color='k',s=50,lw=0,alpha=0.8)

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

    for ax in [ax1,ax2]:
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"galaxy_dist.png"))

def plot_mass_age_radius(catalog,logify=True):

    beagle_galsub = fitsio.getdata("beagle/data/clumps_beagle_output.galsub.fits")

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(9,8),dpi=300,sharex=True)
    fig.subplots_adjust(left=0.09,bottom=0.09,top=0.98,right=0.98,wspace=0.03,hspace=0.03)

    ### Mass vs radius
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_distnorm",y="clump_logMst")
    ax1.errorbar(catalog["clump_distnorm"],catalog[ "clump_logMst"],yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=catalog[ "clump_logMst"],axis=ax1,c="k",nbin=50,fill=True,logify=logify,label="Clumps")

    ### Ages vs radius
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_distnorm",y="clump_logage")
    ax2.errorbar(catalog["clump_distnorm"],catalog["clump_logage"],yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=catalog["clump_logage"],axis=ax2,c="k",nbin=50,fill=True,logify=logify,label="Clumps")

    ### Plotting the mass and ages of the diffuse galaxy light
    xerr,yerr = getBeagleErrors(catalog=beagle_galsub,x="logMst",y="logMst")
    ax2.errorbar(catalog["clump_distnorm"],beagle_galsub["logage"],yerr=yerr,
                    color="tab:red",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=beagle_galsub["logage"],axis=ax2,c="tab:red",nbin=50,fill=True,logify=logify,label="Underlying galaxy")

    ax1.set_ylim(3.8,9.2)
    ax1.set_ylabel("log M$^\\star_{clump}$ [M$_\\odot$]",fontsize=24)
    ax2.set_ylim(5.5,10.5)
    ax2.set_ylabel("log age [yr]",fontsize=24)
    ax2.set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=24)

    for ax in [ax1,ax2]:
        if logify:
            ax.set_xlim(1.5e-1,1e1)
            ax.set_xscale('log')
        else:
            ax.set_xlim(0,5.5)
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    handles, labels = ax2.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    leg = ax2.legend(handles,labels,loc="best",fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.8)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_color())
        txt.set_fontweight(600)
        hndl.set_visible(False)

    savename = "clump_mass_age_radius.png"
    if not logify: savename = savename.replace(".png","_nonlog.png")
    fig.savefig(os.path.join(savedir,savename))

def plot_ssfr_radius(catalog,logify=True):

    beagle_galsub = fitsio.getdata("beagle/data/clumps_beagle_output.galsub.fits")

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=300,tight_layout=True)

    ### sSFR vs radius
    yerr1,yerr2 = getBeagleErrors(catalog=catalog,x="clump_logSFR",y="clump_logMst")
    yerr = np.sqrt(yerr1**2+yerr2**2)
    sSFR = catalog["clump_logSFR"] - catalog["clump_logMst"]
    ax.errorbar(catalog["clump_distnorm"],sSFR,yerr=yerr,
                    color="k",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=sSFR,axis=ax,c="k",nbin=50,fill=True,logify=logify,label="Clumps")

    ### Plotting the mass and ages of the diffuse galaxy light
    yerr1,yerr2 = getBeagleErrors(catalog=beagle_galsub,x="logSFR",y="logMst")
    yerr = np.sqrt(yerr1**2+yerr2**2)
    sSFR = beagle_galsub["logSFR"] - beagle_galsub["logMst"]
    ax.errorbar(catalog["clump_distnorm"],sSFR,yerr=yerr,
                    color="tab:red",marker='o',markersize=4,mew=0,lw=0,ecolor="k",elinewidth=0.8,capsize=0,alpha=0.25,zorder=8)
    plot_median(xdata=catalog["clump_distnorm"],ydata=sSFR,axis=ax,c="tab:red",nbin=50,fill=True,logify=logify,label="Underlying galaxy")

    ax.set_ylim(-12.5,-5.5)
    ax.set_ylabel("log sSFR [yr$^{-1}$]",fontsize=24)
    ax.set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=24)

    if logify:
        ax.set_xlim(1.5e-1,1e1)
        ax.set_xscale('log')
    else:
        ax.set_xlim(0,5.5)
    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    leg = ax.legend(handles,labels,loc="best",fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.8)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_color())
        txt.set_fontweight(600)
        hndl.set_visible(False)

    savename = "clump_ssfr_radius.png"
    if not logify: savename = savename.replace(".png","_nonlog.png")
    fig.savefig(os.path.join(savedir,savename))

def plot_age_radius(catalog,logify=True):

    clump_flux,gal_flux = {},{}
    for filt in "ugriz":
        clump_flux[filt] = catalog["clump_flux_%s"%filt]
        gal_flux[filt]   = catalog["gal_flux_%s"%filt]
    fracL = np.log10(clump_flux["u"]/gal_flux["u"])

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(9,12),dpi=300,sharex=True,tight_layout=True)

    im1 = ax1.scatter(catalog["clump_distnorm"],catalog["clump_logage"],c=fracL,vmin=-4,vmax=-0.5,cmap=plt.cm.inferno_r,s=8,lw=0,alpha=0.7)
    fracL_arr = np.array([-4,-2.2,-1.8,-1.5,-0.5])
    norm = Normalize(vmin=-4,vmax=-0.5)
    for fracL0,fracL1 in zip(fracL_arr[:-1],fracL_arr[1:]):
        cond = (fracL0<fracL) & (fracL<fracL1)
        color = plt.cm.inferno_r(norm((fracL1+fracL0)/2))
        qlabel = "log($f_u$)"
        if fracL0==fracL_arr[0]: label = "{:s}<{:.1f}".format(qlabel,fracL1)
        elif fracL1==fracL_arr[-1]: label = "{:.1f}<{:s}".format(fracL0,qlabel)
        else: label = "{:.1f}<{:s}<{:.1f}".format(fracL0,qlabel,fracL1)
        plot_median(xdata=catalog["clump_distnorm"][cond],ydata=catalog["clump_logage"][cond],axis=ax1,c=color,nbin=20,alpha=1,logify=logify,label=label)

    mass = catalog["gal_logMst"]
    im2 = ax2.scatter(catalog["clump_distnorm"],catalog["clump_logage"],c=mass,vmin=6.5,vmax=11,cmap=plt.cm.inferno_r,s=8,lw=0,alpha=0.7)
    mass_arr = np.array([6,9,9.8,12])
    norm = Normalize(vmin=6.5,vmax=11)
    for mass0,mass1 in zip(mass_arr[:-1],mass_arr[1:]):
        cond = (mass0<mass) & (mass<mass1)
        color = plt.cm.inferno_r(norm((mass1+mass0)/2))
        qlabel = "log(M$^\\star_{gal}$)"
        if mass0==mass_arr[0]: label = "{:s}<{:.1f}".format(qlabel,mass1)
        elif mass1==mass_arr[-1]: label = "{:.1f}<{:s}".format(mass0,qlabel)
        else: label = "{:.1f}<{:s}<{:.1f}".format(mass0,qlabel,mass1)
        plot_median(xdata=catalog["clump_distnorm"][cond],ydata=catalog["clump_logage"][cond],axis=ax2,c=color,nbin=25,alpha=1,logify=logify,label=label)

    zred = catalog["gal_redshiftdr14"]
    im3 = ax3.scatter(catalog["clump_distnorm"],catalog["clump_logage"],c=zred,vmin=0,vmax=0.06,cmap=plt.cm.inferno,s=8,lw=0,alpha=0.7)
    zred_arr = np.array([0,0.02,0.04,0.06])
    norm = Normalize(vmin=0,vmax=0.06)
    for zred0,zred1 in zip(zred_arr[:-1],zred_arr[1:]):
        cond = (zred0<zred) & (zred<zred1)
        color = plt.cm.inferno(norm((zred1+zred0)/2))
        qlabel = "$z$"
        label = "{:.2f}<{:s}<{:.2f}".format(zred0,qlabel,zred1)
        plot_median(xdata=catalog["clump_distnorm"][cond],ydata=catalog["clump_logage"][cond],axis=ax3,c=color,nbin=25,alpha=1,logify=logify,label=label)

    ax3.set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=24)

    for ax,im,clabel in zip([ax1,ax2,ax3],[im1,im2,im3],["log($f_{clump,U}/f_{gal,U}$)","log($M^\\star_{gal}$)","Redshift, $z$"]):

        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="4%", pad=0.1)
        cbax = fig.colorbar(mappable=im, cax=cbaxes, orientation="vertical")
        cbax.ax.set_ylabel(clabel, fontsize=24)
        cbax.ax.tick_params(labelsize=16)

        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] for h in handles]
        leg = ax.legend(handles,labels,loc=4,fontsize=13,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.8)
        for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
            txt.set_color(hndl.get_color())
            txt.set_alpha(1)
            txt.set_fontweight(600)
            hndl.set_visible(False)

        if logify:
            # ax.set_xlim(3e-1,1.5e1)
            ax.set_xlim(1.5e-1,1.3e1)
            ax.set_xscale('log')
        else:
            ax.set_xlim(0,5.8)

        ax.set_ylim(5.5,10.5)
        ax.set_ylabel("log Clump age [yr]",fontsize=24)
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(14) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    savename = "clump_age_radius.png"
    if not logify: savename = savename.replace(".png","_nonlog.png")
    fig.savefig(os.path.join(savedir,savename))

def plot_SFMS(catalog,guo_cat):

    fig,ax = plt.subplots(1,1,figsize=(11,8),dpi=300,tight_layout=True)
    condz = (0.03<catalog["gal_redshiftdr14"]) & (catalog["gal_redshiftdr14"]<0.05)

    ## Plot S82 Clumps+galaxies
    uniq_gals,iuniq = np.unique(catalog["gzh_id"],return_index=True)
    xerr,yerr = getBeagleErrors(catalog=catalog,x="clump_logMst",y="clump_logSFR")
    ax.errorbar(catalog["clump_logMst"][ condz],catalog["clump_logSFR"][ condz],xerr=xerr[:, condz],yerr=yerr[:, condz],
                    color="k",ecolor="k",marker='o',markersize=6,mew=0,lw=0,elinewidth=0.8,capsize=0,alpha=0.8,zorder=10,label="S82 Clumps (0.03<z<0.05)")
    ax.errorbar(catalog["clump_logMst"][~condz],catalog["clump_logSFR"][~condz],xerr=xerr[:,~condz],yerr=yerr[:,~condz],
                    color="gray",ecolor="gray",marker='o',markersize=6,mew=0,lw=0,elinewidth=0.8,capsize=0,alpha=0.8,zorder=10,label="S82 Clumps")
    xerr,yerr = getBeagleErrors(catalog=catalog[iuniq],x="gal_logMst",y="gal_logSFR")
    ax.errorbar(catalog["gal_logMst"][iuniq],catalog["gal_logSFR"][iuniq],xerr=xerr,yerr=yerr,
                    color="tab:red",marker='*',markersize=15,mew=0,lw=0,alpha=0.9,zorder=9,label="S82 Host galaxies")

    cond = catalog["gal_logSFR"] - catalog["gal_logMst"] < -11
    print(np.unique(catalog["gzh_id"][cond]))

    ## Plot Guo+18 galaxies
    uniq_gals,iuniq = np.unique(guo_cat["GalID"],return_index=True)
    ax.errorbar(guo_cat["CMass"],guo_cat["CSFR"],color="tab:green",marker='o',markersize=4,mew=0,lw=0,alpha=0.5,zorder=5,label="Guo+18 clumps")
    ax.errorbar(guo_cat["logM*"][iuniq],guo_cat["logSFR"][iuniq],color=plt.cm.Set1(0.4),marker='*',markersize=8,mew=0,lw=0,alpha=0.5,zorder=4,label="Guo+18 Host galaxies")

    ## Plot SDSS galaxies
    data = fitsio.getdata("catalogs/sdss/SDSS_DR7_gal_info_v5_2.fits")
    mass = fitsio.getdata("catalogs/sdss/SDSS_DR7_gal_totlgm_v5_2.fits")["AVG"]
    sfr  = fitsio.getdata("catalogs/sdss/SDSS_DR7_gal_totsfr_v5_2.fits")["AVG"]
    cond = (data["Z"] < 0.5) & (data["SN_MEDIAN"] > 5) & (data["targettype"]=="GALAXY") & (data["subclass"]=="STARFORMING") & (data["kcor_model_mag"][:,1]<18) & (mass!=-1.) & (sfr!=-99.)
    mass = mass[cond]
    sfr  = sfr[cond]

    N = 6
    dbinx,dbiny = 0.05,0.05
    binsx = np.arange(  0,20,dbinx)
    binsy = np.arange(-10,10,dbiny)
    bincx = 0.5*(binsx[1:]+binsx[:-1])
    bincy = 0.5*(binsy[1:]+binsy[:-1])
    hist2d = np.histogram2d(mass,sfr,bins=[binsx,binsy])[0]
    hist2d = gaussian_filter(hist2d, 0.2)
    hist2d = np.ma.masked_array(hist2d,mask=hist2d<=0)
    ax.contour(bincx,bincy,hist2d.T,N,colors=["k"]*N,linewidths=np.linspace(0.8,2,N),alpha=0.8)

    ## Plot sSFR contours
    ssfr = np.array([-6,-7.,-8.,-9.,-10.,-11.])
    mass = np.arange(0,20,0.01)
    for i,_ssfr in enumerate(ssfr):
        sfr = mass + _ssfr
        ax.plot(mass,sfr,c='k',ls='--',lw=0.5)
        idx = np.argmin(np.abs(sfr-2.6))
        idy = np.argmin(np.abs(mass-11.4))
        _id = min([idx,idy])
        text = ("sSFR = " if i==0 else "") + "10$^{%i}$ yr$^{-1}$"%_ssfr
        ax.text(mass[_id],sfr[_id],text,ha='center',va='center',fontsize=12.5)

    ## Axes decorations
    ax.set_xlabel("log M$^\\star$ [M$_\\odot$]",fontsize=24)
    ax.set_ylabel("log SFR [M$_\\odot$ yr$^{-1}$]",fontsize=24)
    ax.set_xlim(3.8,11.8)
    ax.set_ylim(-4.3,2.8)

    leg = ax.legend(loc=2,fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.6)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        try:    color = hndl.get_color()
        except: color = hndl.get_facecolor()
        txt.set_color(color[0] if isinstance(color,np.ndarray) else color)
        txt.set_fontweight(600)
        txt.set_alpha(1)
        hndl.set_visible(False)

    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"clump_sfms.png"))

def plot_mass_color(catalog,guo_cat):

    _,uid = np.unique(catalog.gzh_id,return_index=True)

    fig,ax = plt.subplots(1,1,figsize=(10,9),dpi=300,tight_layout=True)

    clump_mass = catalog["clump_logMst"]
    clump_col  = catalog["clump_mag_u"]-catalog["clump_mag_r"]
    condz = (0.03<catalog["gal_redshiftdr14"]) & (catalog["gal_redshiftdr14"]<0.05)
    ax.scatter(clump_mass[ condz],clump_col[ condz],c='k',s=25,lw=0,alpha=0.8,label="S82 clumps (0.03<z<0.05)")
    ax.scatter(clump_mass[~condz],clump_col[~condz],c='gray',s=25,lw=0,alpha=0.8,label="S82 clumps")

    gal_mass = catalog["gal_logMst"][uid]
    gal_col  =(catalog["gal_mag_u"]-catalog["gal_mag_r"])[uid]
    ax.scatter(gal_mass,gal_col,c='tab:red',marker='*',s=150,lw=0,alpha=0.95,label="S82 Host galaxies")

    cond_z1 = (0.5 < guo_cat["z"]) & (guo_cat["z"] < 1.0)
    cond_z2 = (1.0 < guo_cat["z"]) & (guo_cat["z"] < 2.0)
    cond_z3 = (2.0 < guo_cat["z"]) & (guo_cat["z"] < 3.0)
    ax.scatter(guo_cat["CMass"][cond_z1],(-2.5*np.log10(guo_cat["F435W"])+2.5*np.log10(guo_cat["F775W" ]))[cond_z1],c='tab:green',s=10,lw=0,alpha=0.8,label="Guo+18 clumps")
    ax.scatter(guo_cat["CMass"][cond_z2],(-2.5*np.log10(guo_cat["F606W"])+2.5*np.log10(guo_cat["F814W" ]))[cond_z2],c='tab:green',s=10,lw=0,alpha=0.8)
    ax.scatter(guo_cat["CMass"][cond_z3],(-2.5*np.log10(guo_cat["F775W"])+2.5*np.log10(guo_cat["F850LP"]))[cond_z3],c='tab:green',s=10,lw=0,alpha=0.8)

    zz = 0.03
    xx = np.arange(0,20,0.01)
    yy = 0.227*xx - 1.16 - 0.352*zz + (0.79 - 0.02) + 0.3
    ax.plot(xx,yy,c='k',lw=1.5,ls='--')

    data = fitsio.getdata("catalogs/sdss/portsmouth_stellarmass_starforming_salp-26.fits")
    cond = (data["Z"] < 0.06)
    mass = data["LOGMASS"][cond]
    col  = data["MAGSCALED"][cond,0] - data["MAGSCALED"][cond,2]

    N = 4
    dbinx,dbiny = 0.1,0.03
    binsx = np.arange(  0,20,dbinx)
    binsy = np.arange(-10,10,dbiny)
    bincx = 0.5*(binsx[1:]+binsx[:-1])
    bincy = 0.5*(binsy[1:]+binsy[:-1])
    hist2d = np.histogram2d(mass,col,bins=[binsx,binsy])[0]
    hist2d = gaussian_filter(hist2d, 0.2)
    hist2d = np.ma.masked_array(hist2d,mask=hist2d<=0)
    ax.contour(bincx,bincy,hist2d.T,N,colors=["k"]*N,linewidths=np.linspace(0.8,2,N),alpha=0.8)

    ax.set_xlabel("log Mass",fontsize=24)
    ax.set_ylabel("U-R",fontsize=24)
    ax.set_xlim(3.8,11.8)
    ax.set_ylim(-0.8,3.2)

    leg = ax.legend(loc=2,fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.6)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        try:    color = hndl.get_color()
        except: color = hndl.get_facecolor()
        txt.set_color(color[0] if isinstance(color,np.ndarray) else color)
        txt.set_fontweight(600)
        txt.set_alpha(1)
        hndl.set_visible(False)

    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"clump_mass_color.png"))

def plot_frac_light_mass(catalog,guo_cat):

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(16,12),dpi=300,sharey="row")
    fig.subplots_adjust(left=0.07,right=0.98,bottom=0.07,top=0.98,wspace=0.03,hspace=0.15)

    clump_flux,clump_ferr,gal_flux,gal_absM = {},{},{},{}
    for filt in "ugriz":
        clump_flux[filt] = catalog["clump_flux_%s"%filt]
        clump_ferr[filt] = catalog["clump_fluxerr_%s"%filt]
        gal_flux[filt] = catalog["gal_flux_%s"%filt]
        gal_absM[filt] = getAbsMagFromAppMag(catalog["gal_mag_%s"%filt],z=catalog["gal_redshiftdr14"])

    cond = clump_flux["u"]/clump_ferr["u"] > 3
    condz = (0.03<catalog["gal_redshiftdr14"]) & (catalog["gal_redshiftdr14"]<0.05)
    ax1.scatter(gal_absM["u"][cond& condz],(clump_flux["u"]/gal_flux["u"])[cond& condz],c='k',s=25,lw=0,alpha=0.8,zorder=10,label="S82 Clumps (0.03<z<0.05)")
    ax1.scatter(gal_absM["u"][cond&~condz],(clump_flux["u"]/gal_flux["u"])[cond&~condz],c='gray',s=25,lw=0,alpha=0.8,zorder=10,label="S82 Clumps")
    ax1.scatter(guo_cat["GUmag"],guo_cat["fLUV"],c='tab:green',s=15,lw=0,alpha=0.7,zorder=1,label="Guo+18 clumps")
    ax1.axhline(0.08,c='k',ls='--')
    ax1.axhline(0.03,c='k',ls='--')
    ax1.set_yscale("log")
    ax1.set_ylim(3e-4,1.1e0)
    ax1.set_xlim(-13.9,-23.2)
    ax1.set_ylabel("$f_{clump} / f_{gal}$",fontsize=28)
    ax1.set_xlabel("Host galaxy absolute magnitude, $M_{U,gal}$",fontsize=24)

    ax2.scatter(catalog["clump_distnorm"][cond& condz],(clump_flux["u"]/gal_flux["u"])[cond& condz],c='k',s=25,lw=0,alpha=0.8,zorder=10,label="S82 Clumps (0.03<z<0.05)")
    ax2.scatter(catalog["clump_distnorm"][cond&~condz],(clump_flux["u"]/gal_flux["u"])[cond&~condz],c='gray',s=25,lw=0,alpha=0.8,zorder=10,label="S82 Clumps")
    ax2.scatter(guo_cat["GDis"],guo_cat["fLUV"],c='tab:green',s=15,lw=0,alpha=0.7,zorder=1,label="Guo+18 clumps")
    ax2.axhline(0.08,c='k',ls='--')
    ax2.axhline(0.03,c='k',ls='--')
    ax2.set_yscale("log")
    ax2.set_ylim(3e-4,1.1e0)
    ax2.set_xlim(0,5.5)
    ax2.set_xlabel("Norm. Galactocentric Distance, r/r$_{50}$",fontsize=24)

    ax3.scatter(catalog["gal_logMst"][ condz],(catalog["clump_logMst"]-catalog["gal_logMst"])[ condz],c='k',s=25,lw=0,alpha=0.8,zorder=10,label="S82 clumps (0.03<z<0.05)")
    ax3.scatter(catalog["gal_logMst"][~condz],(catalog["clump_logMst"]-catalog["gal_logMst"])[~condz],c='gray',s=25,lw=0,alpha=0.8,zorder=10,label="S82 clumps")
    ax3.scatter(guo_cat["logM*"],guo_cat["CMass"]-guo_cat["logM*"],c="tab:green",s=15,lw=0,alpha=0.8,zorder=1,label="Guo+18 clumps")
    ax3.set_xlabel("log Host galaxy mass [M$_\\odot$]",fontsize=24)
    ax3.set_ylabel("log (M$^\\star_{clump}$/M$^\\star_{gal}$)",fontsize=28)
    ax3.set_xlim(7.2,11.45)
    ax3.set_ylim(-6.2,0.2)

    ax4.scatter(catalog["clump_distnorm"][ condz],(catalog["clump_logMst"]-catalog["gal_logMst"])[ condz],c='k',s=25,lw=0,alpha=0.8,zorder=10,label="S82 clumps (0.03<z<0.05)")
    ax4.scatter(catalog["clump_distnorm"][~condz],(catalog["clump_logMst"]-catalog["gal_logMst"])[~condz],c='gray',s=25,lw=0,alpha=0.8,zorder=10,label="S82 clumps")
    ax4.scatter(guo_cat["GDis"],guo_cat["CMass"]-guo_cat["logM*"],c="tab:green",s=15,lw=0,alpha=0.8,zorder=1,label="Guo+18 clumps")
    ax4.set_xlabel("Norm. Galactocentric Distance, r/r$_{50}$",fontsize=24)
    ax4.set_xlim(0,5.5)
    ax4.set_ylim(-6.2,0.2)

    leg = ax4.legend(loc=1,fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.6)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor()[0])
        txt.set_fontweight(600)
        txt.set_alpha(1)
        hndl.set_visible(False)

    for ax in [ax1,ax2,ax3,ax4]:
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"clump_frac_light_mass.png"))

def plot_totfrac_light_mass(catalog,guo_cat):

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,14),dpi=300,tight_layout=True)

    uniq_gals_S82,iuniq_S82 = np.unique(catalog["gzh_id"],return_index=True)
    uniq_gals_guo,iuniq_guo = np.unique(guo_cat["GalID"],return_index=True)
    condz = (0.03<catalog["gal_redshiftdr14"][iuniq_S82]) & (catalog["gal_redshiftdr14"][iuniq_S82]<0.05)

    clump_flux,gal_flux,gal_absM = {},{},{}
    for filt in "ugriz":
        clump_flux[filt] = catalog["clump_flux_%s"%filt]
        gal_flux[filt] = catalog["gal_flux_%s"%filt]
        gal_absM[filt] = getAbsMagFromAppMag(catalog["gal_mag_%s"%filt],z=catalog["gal_redshiftdr14"])

    totfrac_S82 = np.array([np.sum(clump_flux["u"][catalog["gzh_id"]==galID]) for galID in uniq_gals_S82]) / gal_flux["u"][iuniq_S82]
    totfrac_guo = np.array([np.sum(guo_cat["fLUV"][guo_cat["GalID"]==galID]) for galID in uniq_gals_guo])
    ax1.scatter(gal_absM["u"][iuniq_S82][ condz],totfrac_S82[ condz],c='k',s=50,lw=0,alpha=0.8,zorder=10,label="S82 Clumps (0.03<z<0.05)")
    ax1.scatter(gal_absM["u"][iuniq_S82][~condz],totfrac_S82[~condz],c='gray',s=50,lw=0,alpha=0.8,zorder=10,label="S82 Clumps")
    ax1.scatter(guo_cat["GUmag"][iuniq_guo],totfrac_guo,c='tab:green',s=15,lw=0,alpha=0.7,zorder=1,label="Guo+18 clumps")
    ax1.axhline(0.08,c='k',ls='--')
    ax1.axhline(0.03,c='k',ls='--')
    ax1.set_yscale("log")
    ax1.set_ylim(1e-2,2e0)
    ax1.set_xlim(-13.9,-23.2)
    ax1.set_ylabel("Total fraction of light in clumps,\n$\\sum (f_{clump}) / f_{gal}$",fontsize=24)
    ax1.set_xlabel("Host galaxy absolute magnitude, $M_{U,gal}$",fontsize=24)

    totfrac_S82 = np.array([np.log10(np.sum(10**catalog["clump_logMst"][catalog["gzh_id"]==galID])) for galID in uniq_gals_S82]) - catalog["gal_logMst"][iuniq_S82]
    totfrac_guo = np.array([np.log10(np.sum(10**guo_cat["CMass"][guo_cat["GalID"]==galID])) for galID in uniq_gals_guo]) - guo_cat["logM*"][iuniq_guo]
    ax2.scatter(catalog["gal_logMst"][iuniq_S82][ condz],totfrac_S82[ condz],c='k',s=50,lw=0,alpha=0.8,zorder=10,label="S82 clumps (0.03<z<0.05)")
    ax2.scatter(catalog["gal_logMst"][iuniq_S82][~condz],totfrac_S82[~condz],c='gray',s=50,lw=0,alpha=0.8,zorder=10,label="S82 clumps")
    ax2.scatter(guo_cat["logM*"][iuniq_guo],totfrac_guo,c="tab:green",s=15,lw=0,alpha=0.8,zorder=1,label="Guo+18 clumps")
    ax2.set_xlabel("log Host galaxy mass [M$_\\odot$]",fontsize=24)
    ax2.set_ylabel("Total fraction of mass in clumps,\n$\\log{\\left[ \\sum (M^\\star_{clump})/M^\\star_{gal} \\right]}$",fontsize=24)
    ax2.set_xlim(7.2,11.45)
    ax2.set_ylim(-5.2,0.5)

    leg = ax2.legend(loc=3,fontsize=18,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.6)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        txt.set_color(hndl.get_facecolor()[0])
        txt.set_fontweight(600)
        txt.set_alpha(1)
        hndl.set_visible(False)

    for ax in [ax1,ax2]:
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"clump_totfrac_light_mass.png"))

def plot_mass_to_light(catalog,guo_cat):

    fig,ax = plt.subplots(1,1,figsize=(10,12),dpi=300,tight_layout=True)

    divider = make_axes_locatable(ax)
    dax = divider.append_axes("top", size="30%", pad=0.1)
    bins1 = np.arange(-15,15,0.1)
    bins2 = np.arange(-15,15,0.2)

    uniq_gals_S82,iuniq_S82 = np.unique(catalog["gzh_id"],return_index=True)
    condz = (0.03<catalog["gal_redshiftdr14"]) & (catalog["gal_redshiftdr14"]<0.05)
    dist_term = (4*np.pi*Planck15.luminosity_distance(z=catalog["gal_redshiftdr14"]).cgs.value**2) * (1+catalog["gal_redshiftdr14"])

    clump_flux,clump_ferr,clump_lum,gal_flux,gal_lum = {},{},{},{},{}
    for filt in "ugriz":
        clump_flux[filt] = catalog["clump_flux_%s"%filt]
        clump_ferr[filt] = catalog["clump_fluxerr_%s"%filt]
        gal_flux[filt]   = catalog["gal_flux_%s"%filt]
        gal_flux[filt]   = catalog["gal_fluxerr_%s"%filt]
        clump_lum[filt]  = np.log10(clump_flux[filt] * 1e-29 * (light/sdss_filters_pivot[filt]**2) * sdss_filters_width[filt] * dist_term / 4e33)
        clump_lerr[filt] = np.log10(clump_ferr[filt] * 1e-29 * (light/sdss_filters_pivot[filt]**2) * sdss_filters_width[filt] * dist_term / 4e33)
        gal_lum[filt]    = np.log10(gal_flux[filt]   * 1e-29 * (light/sdss_filters_pivot[filt]**2) * sdss_filters_width[filt] * dist_term / 4e33)
        gal_lerr[filt]   = np.log10(gal_ferr[filt]   * 1e-29 * (light/sdss_filters_pivot[filt]**2) * sdss_filters_width[filt] * dist_term / 4e33)

    cond = clump_flux["u"]/clump_ferr["u"] > 3
    ax.scatter((catalog["clump_logMst"] - clump_lum["u"])[cond& condz],catalog["clump_logMst"][cond& condz],color='k',s=25,lw=0,alpha=0.8,zorder=10,label="S82 clumps (0.03<z<0.05)")
    ax.scatter((catalog["clump_logMst"] - clump_lum["u"])[cond&~condz],catalog["clump_logMst"][cond&~condz],color='gray',s=25,lw=0,alpha=0.8,zorder=10,label="S82 clumps")
    ax.scatter((catalog["gal_logMst"] - gal_lum["u"])[iuniq_S82],catalog["gal_logMst"][iuniq_S82],color='tab:red',marker='*',s=100,lw=0,alpha=0.95,zorder=9,label="S82 Host galaxies")

    dax.hist((catalog["clump_logMst"] - clump_lum["u"])[cond],bins=bins1,density=True,color='k',lw=0,alpha=0.4,zorder=5)
    dax.hist((catalog["clump_logMst"] - clump_lum["u"])[cond&condz],bins=bins1,density=True,color='k',lw=4,histtype="step",zorder=10)
    dax.hist((catalog["gal_logMst"] - gal_lum["u"])[iuniq_S82],bins=bins2,density=True,color='tab:red',lw=3,histtype="step",zorder=9)

    cond_z1 = (0.5 < guo_cat["z"]) & (guo_cat["z"] <= 1.0)
    cond_z2 = (1.0 < guo_cat["z"]) & (guo_cat["z"] <= 1.5)
    cond_z3 = (1.5 < guo_cat["z"]) & (guo_cat["z"] <= 2.0)
    cond_z4 = (2.0 < guo_cat["z"]) & (guo_cat["z"] <= 3.0)

    dist_term = (4*np.pi*Planck15.luminosity_distance(z=guo_cat["z"]).cgs.value**2) * (1+guo_cat["z"])
    guo_clump_lum = np.zeros(len(guo_cat),dtype=float)
    guo_clump_lum[cond_z1] = np.log10(guo_cat["F606W" ][cond_z1] * 1e-29 * (light/hst_filters_pivot["F606W" ]**2) * hst_filters_width["F606W" ] * dist_term[cond_z1] / 4e33)
    guo_clump_lum[cond_z2] = np.log10(guo_cat["F775W" ][cond_z2] * 1e-29 * (light/hst_filters_pivot["F775W" ]**2) * hst_filters_width["F775W" ] * dist_term[cond_z2] / 4e33)
    guo_clump_lum[cond_z3] = np.log10(guo_cat["F850LP"][cond_z3] * 1e-29 * (light/hst_filters_pivot["F850LP"]**2) * hst_filters_width["F850LP"] * dist_term[cond_z3] / 4e33)
    guo_clump_lum[cond_z4] = np.log10(guo_cat["F125W" ][cond_z4] * 1e-29 * (light/hst_filters_pivot["F125W" ]**2) * hst_filters_width["F125W" ] * dist_term[cond_z4] / 4e33)
    ax.scatter(guo_cat["CMass"] - guo_clump_lum,guo_cat["CMass"],c='tab:green',s=10,lw=0,alpha=0.8,zorder=4,label="Guo+18 clumps")
    dax.hist(guo_cat["CMass"] - guo_clump_lum,bins=bins1,density=True,color='tab:green',lw=5,histtype="step",zorder=10)

    uniq_gals_guo,iuniq_guo = np.unique(guo_cat["GalID"],return_index=True)
    guo_cat = guo_cat[iuniq_guo]
    guo_phot = fitsio.getdata("catalogs/guo/Guo17_Galaxy_Catalog_Phot.fits")[iuniq_guo]

    cond_z1 = (0.5 < guo_cat["z"]) & (guo_cat["z"] <= 1.0)
    cond_z2 = (1.0 < guo_cat["z"]) & (guo_cat["z"] <= 1.5)
    cond_z3 = (1.5 < guo_cat["z"]) & (guo_cat["z"] <= 2.0)
    cond_z4 = (2.0 < guo_cat["z"]) & (guo_cat["z"] <= 3.0)

    dist_term = (4*np.pi*Planck15.luminosity_distance(z=guo_cat["z"]).cgs.value**2) * (1+guo_cat["z"])
    guo_gal_lum = np.zeros(len(guo_phot),dtype=float)
    guo_gal_lum[cond_z1] = np.log10(guo_phot["FLUX_AUTO_F606W" ][cond_z1] * 1e-29 * (light/hst_filters_pivot["F606W" ]**2) * hst_filters_width["F606W" ] * dist_term[cond_z1] / 4e33)
    guo_gal_lum[cond_z2] = np.log10(guo_phot["FLUX_AUTO_F775W" ][cond_z2] * 1e-29 * (light/hst_filters_pivot["F775W" ]**2) * hst_filters_width["F775W" ] * dist_term[cond_z2] / 4e33)
    guo_gal_lum[cond_z3] = np.log10(guo_phot["FLUX_AUTO_F850LP"][cond_z3] * 1e-29 * (light/hst_filters_pivot["F850LP"]**2) * hst_filters_width["F850LP"] * dist_term[cond_z3] / 4e33)
    guo_gal_lum[cond_z4] = np.log10(guo_phot["FLUX_AUTO_F125W" ][cond_z4] * 1e-29 * (light/hst_filters_pivot["F125W" ]**2) * hst_filters_width["F125W" ] * dist_term[cond_z4] / 4e33)
    ax.scatter(guo_cat["logM*"] - guo_gal_lum,guo_cat["logM*"],color=plt.cm.Set1(0.4),marker='*',s=100,lw=0,alpha=0.8,zorder=4,label="Guo+18 Host galaxies")
    dax.hist(guo_cat["logM*"] - guo_gal_lum,bins=bins2,density=True,color=plt.cm.Set1(0.4),lw=3,histtype="step",zorder=8)

    ax.set_xlabel("log $(M/L_{u})$",fontsize=24)
    ax.set_ylabel("log Stellar Mass [M$_\\odot$]",fontsize=24)
    ax.set_xlim(-2.5,2.9)
    ax.set_ylim(3.8,11.45)
    dax.set_xlim(ax.get_xlim())
    dax.set_ylabel("Norm. Freq.",fontsize=24)

    leg = ax.legend(loc="best",fontsize=16,handlelength=0,handletextpad=0,markerscale=0,framealpha=0.6)
    for txt,hndl in zip(leg.get_texts(),leg.legendHandles):
        try:    color = hndl.get_color()
        except: color = hndl.get_facecolor()
        txt.set_color(color[0] if isinstance(color,np.ndarray) else color)
        txt.set_fontweight(600)
        txt.set_alpha(1)
        hndl.set_visible(False)

    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    dax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]
    [_.set_visible(False) for _ in dax.get_xticklabels()+dax.get_yticklabels()]
    [_.xaxis.set_major_locator(ticker.MultipleLocator(0.5)) for _ in [ax,dax]]

    fig.savefig(os.path.join(savedir,"clump_mass_to_light.png"))

def plot_metallicities(catalog,plot_err=False):

    from metallicity import direct_method_pyneb, indirect_method

    linecat = fitsio.getdata("samples/final_clump_linecat.fits")

    logTe,logOH_dir,Z_dir = np.zeros((3,len(linecat)),dtype=float) - 99
    logOH_ind_upper,logOH_ind_lower = np.zeros((2,len(linecat)),dtype=float) - 99
    err_dir, err_ind = np.zeros((2,2,len(linecat)))
    Niter = 100

    for i,entry in enumerate(linecat):

        print("\rProcessing {:d}/{:d} ... ".format(i+1,len(linecat)),end="")

        logTe[i], logOH_dir[i], Z_dir[i] = direct_method_pyneb(entry)
        if plot_err and logOH_dir[i]!=-99 and np.isfinite(logOH_dir[i]):
            seeds = (np.random.rand(Niter)*1e8).astype(int)
            _logTe,_logOH_dir,_Z_dir = map(np.array,zip(*Parallel(n_jobs=15,verbose=0,backend="multiprocessing")(delayed(direct_method_pyneb)(copy.deepcopy(entry),simulate_noise=True,seed=seeds[k]) for k in range(Niter))))
            err_dir[:,i] = np.percentile(_logOH_dir,[16,84])
            err_dir[:,i] = np.abs(logOH_dir[i] - err_dir[:,i])

        logOH_ind_upper[i], logOH_ind_lower[i] = indirect_method(entry,calib="P05")
        if plot_err and  logOH_ind_lower[i]!=-99 and np.isfinite(logOH_ind_lower[i]):
            seeds = (np.random.rand(Niter)*1e8).astype(int)
            _logOH_ind_upper,_logOH_ind_lower = map(np.array,zip(*Parallel(n_jobs=15,verbose=0,backend="multiprocessing")(delayed(indirect_method)(copy.deepcopy(entry),simulate_noise=True,seed=seeds[k],calib="P05") for k in range(Niter))))
            err_ind[:,i] = np.percentile(_logOH_ind_lower,[16,84])
            err_ind[:,i] = np.abs(logOH_ind_lower[i] - err_ind[:,i])

    print("done.")

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=300,tight_layout=True)

    cond_dir = (logOH_dir!=-99) & np.isfinite(logOH_dir)
    cond_ind = (logOH_ind_lower!=-99) & (~cond_dir)
    ax.scatter(catalog["clump_distnorm"][cond_dir],logOH_dir[cond_dir],marker='o',color='gray',s=75,lw=0,alpha=0.9)
    ax.scatter(catalog["clump_distnorm"][cond_ind],logOH_ind_lower[cond_ind],marker='x',color='gray',s=40,lw=1.5,alpha=0.9)

    ax.errorbar(catalog["clump_distnorm"][cond_dir],logOH_dir[cond_dir],yerr=err_dir[:,cond_dir],marker='o',color='gray',ecolor='gray',markersize=0,elinewidth=0.8,lw=0,alpha=0.9)
    ax.errorbar(catalog["clump_distnorm"][cond_ind],logOH_ind_lower[cond_ind],yerr=err_ind[:,cond_ind],marker='x',color='gray',ecolor='gray',markersize=0,elinewidth=0.8,lw=0,alpha=0.9)

    metal = np.zeros(len(catalog)) - 99.
    metal[cond_dir] = logOH_dir[cond_dir]
    metal[cond_ind] = logOH_ind_lower[cond_ind]
    cond = (metal!=-99)
    plot_median(xdata=catalog["clump_distnorm"][cond],ydata=metal[cond],axis=ax,c="k",nbin=10,logify=True)

    ax.set_xlabel("Norm. distance [r/r$_{50}$]",fontsize=24)
    ax.set_ylabel("log OH",fontsize=24)
    ax.set_ylim(6.7,9)
    ax.set_xlim(1.5e-1,1e1)
    ax.set_xscale('log')

    OH_sol = 8.72
    dax = ax.twinx()
    dax.set_ylim([x-OH_sol for x in ax.get_ylim()])
    dax.set_ylabel("log(Z/Z$_\\odot$)",fontsize=24)

    ax.scatter(-99,-99,color='k',marker='o',s=75,lw=0,label="Direct Method")
    ax.scatter(-99,-99,color='k',marker='x',s=20,lw=1.5,label="Indirect Method")
    ax.legend(loc=1,fontsize=18,framealpha=0.8)

    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()+dax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"clump_metallicity.png"))

def plot_seeing_sizes(catalog,guo_cat):

    uniq_gals,iuniq = np.unique(catalog["gzh_id"],return_index=True)
    uniq_gals_S82 = catalog[iuniq]
    zred_S82 = uniq_gals_S82["gal_redshiftdr14"]
    seeing_S82 = uniq_gals_S82["gzh_psf_width_r"] / Planck15.arcsec_per_kpc_proper(zred_S82).value
    # seeing_S82 = uniq_gals_S82["clump_psf_avg"] / Planck15.arcsec_per_kpc_proper(zred_S82).value

    # uniq_gals,iuniq = np.unique(guo_cat["GalID"],return_index=True)
    # uniq_gals_guo = guo_cat[iuniq]
    # zred_guo = uniq_gals_guo["z"]
    # seeing_guo = 0.10 / Planck15.arcsec_per_kpc_proper(zred_guo).value

    fig,ax = plt.subplots(1,1,figsize=(10,6),dpi=300,tight_layout=True)

    ax.scatter(zred_S82,seeing_S82,c='k',s=25,lw=0,alpha=0.8)

    dax = ax.twiny()
    zz = np.arange(0.5,3+1e-8,0.01)
    dax.fill_between(zz,0.10/Planck15.arcsec_per_kpc_proper(zz).value,
                        0.13/Planck15.arcsec_per_kpc_proper(zz).value,
                        color="tab:green",lw=0,alpha=0.4)
    dax.set_xlim(0.5,3)
    dax.set_xlabel("Redshift, z",color="tab:green",fontsize=24)
    dax.spines["top"].set_edgecolor("tab:green")
    dax.xaxis.label.set_color("tab:green")
    dax.tick_params(axis='x', colors="tab:green")

    ax.set_ylim(0,1.7)
    ax.set_xlim(0,0.065)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Seeing Resolution [kpc]",fontsize=24)
    ax.set_xlabel('Redshift, z',fontsize=24)
    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()+dax.get_xticklabels()]

    fig.savefig(os.path.join(savedir,"clump_seeing.png"))

def get_Nclump_cmap():

    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist = cmaplist[::22][::-1]
    cmaplist[-1] = (0.0, 0.0, 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(cmaplist))
    bounds = np.linspace(0.5, 11.5, 12)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap,norm,bounds

def plot_Nclump_colorbar(ax,cmap,norm,bounds):

    from matplotlib.colorbar import ColorbarBase
    ticks = np.arange(11)+1
    label = [str(x+1) for x in range(10)]+["10+"]
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="4%", pad=0.1)
    cbax = ColorbarBase(cbaxes, cmap=cmap, norm=norm, spacing='proportional', ticks=ticks, boundaries=bounds)
    cbax.ax.set_yticklabels(label)
    cbax.ax.tick_params(labelsize=16)
    cbax.ax.set_ylabel("No. of clumps per galaxy", fontsize=24)

def mk_xbrokenaxes(ax1,ax2,xlims):

    ax1.set_xlim(xlims[0])
    ax2.set_ylim(xlims[1])
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(axis="both",which="both",right=False,labelright=False)
    ax2.tick_params(axis="both",which="both",left=False,labelleft=False)

    d = 0.008
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d,1+d),(-d,+d),**kwargs)
    ax1.plot((1-d,1+d),(1-d,1+d),**kwargs)
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d,+d),(-d,+d),**kwargs)
    ax2.plot((-d,+d),(1-d,1+d),**kwargs)

def plot_surf_dens(catalog,guo_cat):

    uniq_gals,iuniq = np.unique(catalog["gzh_id"],return_index=True)
    NClumps_S82 = np.array([sum(catalog["gzh_id"]==x) for x in uniq_gals])
    uniq_gals_S82 = catalog[iuniq]
    zred_S82 = uniq_gals_S82["gal_redshiftdr14"]
    size_S82 = uniq_gals_S82["gal_reff"] / Planck15.arcsec_per_kpc_proper(zred_S82).value

    uniq_gals,iuniq = np.unique(guo_cat["GalID"],return_index=True)
    NClumps_guo = np.array([sum(guo_cat["GalID"]==x) for x in uniq_gals])
    uniq_gals_guo = guo_cat[iuniq]
    zred_guo = uniq_gals_guo["z"]
    size_guo = uniq_gals_guo["MajAxis"] / Planck15.arcsec_per_kpc_proper(zred_guo).value

    ##############

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(20,12),sharey="row",dpi=300,tight_layout=False)
    fig.subplots_adjust(left=0.06,right=0.95,bottom=0.07,top=0.97,hspace=0.18,wspace=0.03)

    cmap,norm,bounds = get_Nclump_cmap()

    ax1.scatter(zred_S82,size_S82,c=NClumps_S82,s=5+10*NClumps_S82,norm=norm,cmap=cmap)
    ax2.scatter(zred_guo,size_guo,c=NClumps_guo,s=5+10*NClumps_guo,norm=norm,cmap=cmap)
    ax1.set_title("S82 Clumps",va="top",ha="center",fontsize=24,fontweight=600)
    ax2.set_title("Guo+18 Clumps",va="top",ha="center",fontsize=24,fontweight=600)

    mk_xbrokenaxes(ax1,ax2,xlims=[(0.0,0.063),(0.45,3.05)])
    ax1.set_ylim(0,12.2)
    ax1.set_ylabel("Galaxy size [kpc]",fontsize=22)
    fig.text(0.5,0.515,'Redshift, z',ha='center',va='center',fontsize=24)

    ax3.scatter(zred_S82,NClumps_S82/(np.pi*size_S82**2),c=NClumps_S82,s=5+10*NClumps_S82,norm=norm,cmap=cmap)
    ax4.scatter(zred_guo,NClumps_guo/(np.pi*size_guo**2),c=NClumps_guo,s=5+10*NClumps_guo,norm=norm,cmap=cmap)
    ax3.set_title("S82 Clumps",va="top",ha="center",fontsize=24,fontweight=600)
    ax4.set_title("Guo+18 Clumps",va="top",ha="center",fontsize=24,fontweight=600)

    mk_xbrokenaxes(ax3,ax4,xlims=[(0.0,0.063),(0.45,3.05)])
    ax3.set_ylabel("Clump Number Surface Density [kpc$^{-2}$]",fontsize=22)
    ax3.set_ylim(3e-3,5e0)
    ax3.set_yscale("log")
    fig.text(0.5,0.03,'Redshift, z',ha="center",va="center",fontsize=24)

    plot_Nclump_colorbar(ax=ax2,cmap=cmap,norm=norm,bounds=bounds)
    plot_Nclump_colorbar(ax=ax4,cmap=cmap,norm=norm,bounds=bounds)

    for ax in [ax1,ax2,ax3,ax4]:
        ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
        [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"clump_surf_dens.png"))

def plot_clump_ages(catalog,guo_cat):

    fig,ax = plt.subplots(1,1,figsize=(12,7),dpi=300,tight_layout=True)

    uniq_gals,iuniq = np.unique(catalog["gzh_id"],return_index=True)
    gal_mass = catalog["gal_logMst"][iuniq]

    N_clumps = np.zeros(len(uniq_gals))
    min_ages, max_ages, avg_ages = np.zeros((3,len(uniq_gals)))
    for i,galID in enumerate(uniq_gals):
        cond = (catalog["gzh_id"]==galID)
        N_clumps[i] = sum(cond)
        min_ages[i] = np.min(catalog["clump_logage"][cond])
        max_ages[i] = np.max(catalog["clump_logage"][cond])
        avg_ages[i] = np.mean(catalog["clump_logage"][cond])

    cmap,norm,bounds = get_Nclump_cmap()
    plot_Nclump_colorbar(ax=ax,cmap=cmap,norm=norm,bounds=bounds)

    ax.scatter(gal_mass,avg_ages,s=10+10*N_clumps,c=N_clumps,
                    cmap=cmap,norm=norm,lw=0,edgecolor='k',alpha=0.9)
    ax.errorbar(gal_mass,avg_ages,
                    yerr=[avg_ages-min_ages,max_ages-avg_ages],
                    color='gray',marker='o',markersize=0,mew=0,lw=0,
                    elinewidth=0.8,capsize=0,alpha=0.6)
    for i,galID in enumerate(uniq_gals):
        cond = (catalog["gzh_id"]==galID)
        ax.scatter([gal_mass[i]]*sum(cond),catalog["clump_logage"][cond],c='k',marker='x',s=10,lw=1,alpha=0.6)

    cond = (gal_mass>0)
    plot_median(xdata=gal_mass[cond],ydata=avg_ages[cond],axis=ax,c="k",logify=False,nbin=20)

    ax.set_xlim(7.2,11.45)
    ax.set_xlabel("log Galaxy stellar mass [M$_\\odot$]",fontsize=24)
    ax.set_ylim(5.8,10.2)
    ax.set_ylabel("log Clump age [yr]",fontsize=24)

    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"clump_age_galmass.png"))

def plot_clump_colors(catalog,guo_cat):

    fig,ax = plt.subplots(1,1,figsize=(12,7),dpi=300,tight_layout=True)

    uniq_gals,iuniq = np.unique(catalog["gzh_id"],return_index=True)
    gal_mass = catalog["gal_logMst"][iuniq]

    cond_col = (catalog["clump_flux_u"]/catalog["clump_fluxerr_u"] > 3) & (catalog["clump_flux_r"]/catalog["clump_fluxerr_r"] > 3)
    N_clumps = np.zeros(len(uniq_gals))
    min_color, max_color, avg_color = np.zeros((3,len(uniq_gals)))
    for i,galID in enumerate(uniq_gals):
        cond = (catalog["gzh_id"]==galID)
        N_clumps[i] = sum(cond)
        min_color[i] = np.min((catalog["clump_mag_u"]-catalog["clump_mag_r"])[cond & cond_col])
        max_color[i] = np.max((catalog["clump_mag_u"]-catalog["clump_mag_r"])[cond & cond_col])
        avg_color[i] = np.mean((catalog["clump_mag_u"]-catalog["clump_mag_r"])[cond & cond_col])

    cmap,norm,bounds = get_Nclump_cmap()
    plot_Nclump_colorbar(ax=ax,cmap=cmap,norm=norm,bounds=bounds)

    ax.scatter(gal_mass,avg_color,s=10+10*N_clumps,c=N_clumps,
                    cmap=cmap,norm=norm,lw=0.6,edgecolor='k',alpha=0.9)
    ax.errorbar(gal_mass,avg_color,
                    yerr=[avg_color-min_color,max_color-avg_color],
                    color='gray',marker='o',markersize=0,mew=0,lw=0,
                    elinewidth=0.8,capsize=0,alpha=0.6)
    for i,galID in enumerate(uniq_gals):
        cond = (catalog["gzh_id"]==galID)
        ax.scatter([gal_mass[i]]*sum(cond&cond_col),
                   (catalog["clump_mag_u"]-catalog["clump_mag_r"])[cond&cond_col],c='k',marker='x',s=10,lw=1,alpha=0.6)

    cond = (gal_mass>0)
    plot_median(xdata=gal_mass[cond],ydata=avg_color[cond],axis=ax,c="k",logify=False,nbin=20,s=0)

    ax.set_xlim(7.2,11.45)
    ax.set_xlabel("log Galaxy stellar mass [M$_\\odot$]",fontsize=24)
    ax.set_ylim(-1.2,2.4)
    ax.set_ylabel("(U-R)$_{clump}$",fontsize=24)

    ax.grid(linestyle=':', linewidth=0.5, color='k', which="both")
    [_.set_fontsize(16) for _ in ax.get_xticklabels()+ax.get_yticklabels()]

    fig.savefig(os.path.join(savedir,"clump_color_galmass.png"))

if __name__ == '__main__':

    savedir = "plots/final/"
    catalog = fitsio.getdata("samples/final_clump_catalog.fits")

    guo_cat = fitsio.getdata("catalogs/guo/Guo17_Clumps_Catalog.fits")
    guocond = (guo_cat["f_zphot"]!=1) & (guo_cat["VRFlag"]!=1) & (guo_cat["BMFlag"]!=1) & (guo_cat["EFlag"]!=1)
    guo_cat = guo_cat[guocond]
    guo_cat["logA"] = guo_cat["logA"] + 9

    # plot_pretty_stamps(catalog,jpg="S82")
    # plot_pretty_stamps(catalog,jpg="DR14")
    # plot_mass_redshift_dist(catalog,guo_cat)
    plot_mass_age_radius(catalog,logify=True)
    # plot_ssfr_radius(catalog,logify=True)
    # plot_age_radius(catalog,logify=True)
    # plot_SFMS(catalog,guo_cat)
    # plot_mass_color(catalog,guo_cat)
    # plot_frac_light_mass(catalog,guo_cat)
    # plot_totfrac_light_mass(catalog,guo_cat)
    # plot_mass_to_light(catalog,guo_cat)
    # plot_metallicities(catalog,plot_err=True)

    # plot_seeing_sizes(catalog,guo_cat)
    # plot_surf_dens(catalog,guo_cat)
    # plot_clump_ages(catalog,guo_cat)
    # plot_clump_colors(catalog,guo_cat)

    # plt.show()

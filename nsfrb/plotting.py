import matplotlib
from scipy.interpolate import interp1d
from nsfrb.outputlogging import printlog
import matplotlib.animation as animation
from nsfrb import imaging
from nsfrb import pipeline
from dsamfs import utils as pu
from astropy.time import Time
from astropy import units as u
from nsfrb.planning import nvss_cat,atnf_cat,find_fast_vis_label
from nsfrb.config import tsamp_slow,tsamp,CH0,CH_WIDTH , AVERAGING_FACTOR,nsamps,NUM_CHANNELS,fmin,fmax,tsamp_imgdiff,T
import time
matplotlib.use('agg')
import matplotlib.pyplot as plt
fsize=45
fsize2=35
plt.rcParams.update({
                    'font.size': fsize,
                    'font.family': 'sans-serif',
                    'axes.labelsize': fsize,
                    'axes.titlesize': fsize,
                    'xtick.labelsize': fsize,
                    'ytick.labelsize': fsize,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 1,
                    'lines.markersize': 5,
                    'legend.fontsize': fsize2,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})
import numpy as np
import sys
import csv
import os
from nsfrb.config import inject_file,recover_file,img_dir,binary_file,flagged_antennas,bad_antennas,cutterfile
import dsautils.dsa_store as ds
ETCD = ds.DsaStore()

plotting_now = False

def plot_uv_coverage(u, v, title='u-v Coverage'):
    """
    Plot the u-v coverage.
    This function creates a scatter plot of the u-v points and their symmetrical counterparts. It is used to visualize the spatial frequency coverage in radio interferometry.
    Parameters:
    u and v: Arrays of coordinates.
    """
    max_u = max(np.max(u), -np.min(u))
    min_u = min(np.min(u), -np.max(u))
    max_v = max(np.max(v), -np.min(v))
    min_v = min(np.min(v), -np.max(v))

    plt.scatter(u, v, marker='.', color='b')
    plt.scatter(-u, -v, marker='.', color='b')  # Symmetry
    plt.xlim(min_u, max_u)
    plt.ylim(min_v, max_v)
    plt.xlabel('u (m)')
    plt.ylabel('v (m)')
    plt.title(title)
    plt.grid(True)


def plot_amplitude_vs_uv_distance(uv_distance, average_amplitude):
    """
    Plot average amplitude vs. UV distance.
    
    Parameters:
    - uv_distance (array-like): Array of UV distances.
    - average_amplitude (array-like): Array of average amplitudes.
    """
    plt.scatter(uv_distance, average_amplitude, c='r', s=1, alpha=0.5)
    plt.xlabel('UV Distance')
    plt.ylabel('Average Amplitude')
    plt.title('Average Amplitude vs. UV Distance')
    plt.grid(True)


def plot_phase_vs_uv_distance(uv_distance, average_phase):
    """
    Plot average phase vs. UV distance.
    
    Parameters:
    - uv_distance (array-like): Array of UV distances.
    - average_phase (array-like): Array of average phases.
    """
    plt.scatter(uv_distance, average_phase, c='g', s=1, alpha=0.5)
    plt.xlabel('UV Distance')
    plt.ylabel('Average Phase')
    plt.title('Average Phase vs. UV Distance')
    plt.grid(True)


def plot_uv_analysis(u, v, average_amplitude, average_phase, save_to_pdf=False, pdf_filename='plot.pdf'):
    """
    Integrates plotting of UV coverage, average amplitude vs. UV distance, 
    and average phase vs. UV distance into one comprehensive function with subplots.
    """
    uv_distance = np.sqrt(u**2 + v**2)
    plt.figure(figsize=(18, 6))

    # Plot UV coverage
    plt.subplot(1, 3, 1)
    plot_uv_coverage(u, v)
    plt.axis('equal')  # Ensure equal aspect ratio

    # Plot Amplitude vs. UV Distance
    plt.subplot(1, 3, 2)
    plot_amplitude_vs_uv_distance(uv_distance, average_amplitude)

    # Plot Phase vs. UV Distance
    plt.subplot(1, 3, 3)
    plot_phase_vs_uv_distance(uv_distance, average_phase)

    plt.tight_layout()

    if save_to_pdf:
        plt.savefig(pdf_filename)
    else:
        plt.show()


def plot_dirty_images(dirty_images, save_to_pdf=False, pdf_filename='dirty_images.pdf'):
    """
    Plot and save the dirty images.

    Args:
        dirty_images (list): List of dirty images.
        save_to_pdf (bool): Whether to save the plot to a PDF file (default: False).
        pdf_filename (str): The filename of the PDF file (default: 'dirty_images.pdf').

    Returns:
        None
    """
    num_images = len(dirty_images)
    grid_size = int(np.ceil(np.sqrt(num_images)))

    plt.figure(figsize=(grid_size * 4, grid_size * 4))

    for i, img in enumerate(dirty_images):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img, cmap='gray', origin='lower')
        plt.title(f"Image {i+1}")
        plt.colorbar()
        plt.axis('off')

    plt.tight_layout()

    if save_to_pdf:
        plt.savefig(pdf_filename)
    else:
        plt.show()


def search_plots_new(canddict,img,isot,RA_axis,DEC_axis,DM_trials,widthtrials,output_dir,show=True,vmax=1000,vmin=0,s100=100,injection=False,searched_image=None,timeseries=[],uv_diag=None,dec_obs=None,slow=False,imgdiff=False,timeseries_nondm=False,pcanddict=dict(),output_file=cutterfile):
    """
    Makes updated diagnostic plots for search system
    """
    #global plotting_now
    #while plotting_now:
    #    continue
    #plotting_now = True
    print("PLOTTING NOW",str("slow" if slow else ""))
    if slow:
        tsamp_use = tsamp_slow
        plotsuffix = "_slow"
    elif imgdiff:
        tsamp_use = tsamp_imgdiff
        plotsuffix = "_imgdiff"
    else:
        tsamp_use = tsamp
        plotsuffix = ""
    print("search plots started")
    printlog("search plots started",output_file=output_file)
    gridsize = len(RA_axis)
    decs,ras,wids,dms=np.array(canddict['dec_idxs'],dtype=int),np.array(canddict['ra_idxs'],dtype=int),np.array(canddict['wid_idxs'],dtype=int),np.array(canddict['dm_idxs'],dtype=int)#np.unravel_index(np.arange(32*32*2*3)[(imgsearched>2500).flatten()],(32,32,3,2))#[1].shape
    snrs = np.array(canddict['snrs'])#imgsearched.flatten()[(imgsearched>2500).flatten()]
    names = np.array(canddict['names'])
    print("first hurdle passed")
    """
    #check if the candidate is an injection
    injection = False
    with open(inject_file,"r") as csvfile:
        re = csv.reader(csvfile,delimiter=',')
        i = 0
        for row in re:
            if i != 0:
                if row[0] == isot:
                    injection = True
                    break
            i += 1
    csvfile.close()
    """
    plot_period = np.argmax(snrs) in pcanddict.keys()
    fig=plt.figure(figsize=(40,40*((2/3) if imgdiff else 1) + (15 if plot_period else 0)))
    if injection:
        fig.patch.set_facecolor('red')
    elif slow:
        fig.patch.set_facecolor('lightblue')
    elif imgdiff:
        fig.patch.set_facecolor('palegreen')
    nrows = (2 if imgdiff else 3) + (1 if plot_period else 0)
    ncols = 2
    gs = fig.add_gridspec(nrows,ncols)
    ax = fig.add_subplot(gs[0,0])#plt.subplot(3,2,1)

    #ax.imshow((img.mean((2,3)))[:,::-1],cmap='binary',aspect='auto',extent=[np.nanmin(RA_axis),np.nanmax(RA_axis),np.nanmin(DEC_axis),np.nanmax(DEC_axis)])
    #get 2D grids
    if uv_diag is not None and dec_obs is not None:
        print("getting new RA,DEC 2D grid...")
        ra_grid_2D,dec_grid_2D,elev = imaging.uv_to_pix(Time(isot,format='isot').mjd,
                                        len(DEC_axis),DEC=dec_obs,
                                        two_dim=True,manual=False,uv_diag=uv_diag)
        print("done")
    
    if searched_image is not None:
        
        if uv_diag is not None and dec_obs is not None:
            print("making scatter plot...",ra_grid_2D.shape,len(RA_axis))
            ax.scatter(ra_grid_2D[:,-searched_image.shape[1]:].flatten(),dec_grid_2D[:,-searched_image.shape[1]:].flatten(),c=(searched_image.max((2,3))).flatten(),cmap='binary',vmin=vmin/2,vmax=vmax,alpha=0.1)
            print("done")
        else:
            ax.imshow((searched_image.max((2,3)))[:,::-1],cmap='binary',aspect='auto',extent=[np.nanmin(RA_axis),np.nanmax(RA_axis),np.nanmin(DEC_axis),np.nanmax(DEC_axis)],vmin=vmin/2,vmax=vmax)
        #ax.contour(searched_image.max((2,3))[:,::-1],cmap='jet',extent=[np.nanmin(RA_axis),np.nanmax(RA_axis),np.nanmax(DEC_axis),np.nanmin(DEC_axis)],linewidths=4,levels=5)
    else:
        if uv_diag is not None:
            ax.scatter(ra_grid_2D[:,-len(RA_axis):].flatten(),dec_grid_2D[:,-len(RA_axis):].flatten(),c=(img.mean((2,3))).flatten(),cmap='pink_r',alpha=0.1)
        else:
            ax.imshow((img.mean((2,3)))[:,::-1],cmap='pink_r',aspect='auto',extent=[np.nanmin(RA_axis),np.nanmax(RA_axis),np.nanmin(DEC_axis),np.nanmax(DEC_axis)])
    print("done with new stuff")


    printlog("scatter plot done",output_file=output_file)
    if uv_diag is not None and dec_obs is not None:
        ra_grid_2D_cut = ra_grid_2D[:,-searched_image.shape[1]:]
        dec_grid_2D_cut = dec_grid_2D[:,-searched_image.shape[1]:]
        if 'predicts' in canddict.keys():
            ra_grid_2D_cut = ra_grid_2D[:,-searched_image.shape[1]:]
            dec_grid_2D_cut = dec_grid_2D[:,-searched_image.shape[1]:]
            c=ax.scatter(ra_grid_2D_cut[decs[canddict['predicts']==0],ras[canddict['predicts']==0]].flatten(),
                    dec_grid_2D_cut[decs[canddict['predicts']==0],ras[canddict['predicts']==0]].flatten(),
                    c=snrs[canddict['predicts']==0],marker='o',cmap='jet',alpha=0.5,
                    s=3000,#*snrs[canddict['predicts']==0]/s100,
                    vmin=vmin,vmax=vmin*2,linewidths=4,edgecolors='limegreen')
            c=ax.scatter(ra_grid_2D_cut[decs[canddict['predicts']==1],ras[canddict['predicts']==1]].flatten(),
                    dec_grid_2D_cut[decs[canddict['predicts']==1],ras[canddict['predicts']==1]].flatten(),
                    c=snrs[canddict['predicts']==1],marker='o',cmap='jet',alpha=0.5,
                    s=3000,#*snrs[canddict['predicts']==1]/s100,
                    vmin=vmin,vmax=vmin*2,linewidths=4,edgecolors='violet')
        else:
            c=ax.scatter(ra_grid_2D_cut[decs,ras].flatten(),
                    dec_grid_2D_cut[decs,ras].flatten(),c=snrs,marker='o',cmap='jet',alpha=0.5,
                    s=3000,#*snrs/s100,
                    vmin=vmin,vmax=vmin*2)
    else:
        if 'predicts' in canddict.keys():
            c=ax.scatter(RA_axis[ras][canddict['predicts']==0],DEC_axis[decs][canddict['predicts']==0],c=snrs[canddict['predicts']==0],marker='o',cmap='jet',alpha=0.5,
                    s=3000,#100*snrs[canddict['predicts']==0]/s100,
                    vmin=vmin,vmax=vmin*2,linewidths=4,edgecolors='limegreen')
            c=ax.scatter(RA_axis[ras][canddict['predicts']==1],DEC_axis[decs][canddict['predicts']==1],c=snrs[canddict['predicts']==1],marker='o',cmap='jet',alpha=0.5,
                    s=3000,#100*snrs[canddict['predicts']==1]/s100,
                    vmin=vmin,vmax=vmin*2,linewidths=4,edgecolors='violet')
        else:
            c=ax.scatter(RA_axis[ras],DEC_axis[decs],c=snrs,marker='o',cmap='jet',alpha=0.5,
                    s=3000,#100*snrs/s100,
                    vmin=vmin,vmax=vmin*2)#(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
    plt.colorbar(mappable=c,ax=ax,label='S/N')
    #nvss sources
    nvsspos,tmp,tmp = nvss_cat(Time(isot,format='isot').mjd,DEC_axis[len(DEC_axis)//2],sep=np.abs(np.max(DEC_axis)-np.min(DEC_axis))*u.deg)
    ax.plot(nvsspos.ra.value,nvsspos.dec.value,'o',markerfacecolor='none',markeredgecolor='blue',markersize=20,markeredgewidth=4,label='NVSS Source')
    #pulsars
    atnfpos,tmp,tmp,tmp,tmp,tmp = atnf_cat(Time(isot,format='isot').mjd,DEC_axis[len(DEC_axis)//2],sep=np.abs(np.max(DEC_axis)-np.min(DEC_axis))*u.deg)
    ax.plot(atnfpos.ra.value,atnfpos.dec.value,'s',markerfacecolor='none',markeredgecolor='blue',markersize=20,markeredgewidth=4,label='ATNF Pulsar')
    ax.legend(loc='lower right',frameon=True,facecolor='lightgrey')
    ax.axvline(RA_axis[gridsize//2],color='grey')
    ax.axhline(DEC_axis[gridsize//2],color='grey')
    ax.set_xlabel(r"RA ($^\circ$)")
    ax.set_ylabel(r"DEC ($^\circ$)")
    ax.invert_xaxis()
    ax.set_xlim(np.max(RA_axis),np.min(RA_axis))
    ax.set_ylim(np.min(DEC_axis),np.max(DEC_axis))
    printlog("psr plot done",output_file=output_file)

    ax = fig.add_subplot(gs[0,1])#ax=plt.subplot(3,2,2)
    if searched_image is not None and (not imgdiff):
        #plot the DM transform thing for peak candidate
        showidx = np.nanargmax(snrs)
        printlog("SHOWING CAND " + str(showidx) + ", SNR=" + str(snrs[showidx]) + ", DM=" + str(DM_trials[dms][showidx]) + ", WID=" + str(widthtrials[wids][showidx]) + ", RAIDX=" + str(ras[showidx]) + ", DECIDX=" + str(decs[showidx]),output_file=output_file)
        showx,showy,showname = ras[showidx],decs[showidx],names[showidx]
        ax.set_title(showname)

        dmtxinterp1 = np.zeros((len(widthtrials),100))
        for i in range(len(widthtrials)):
            finterp = interp1d(DM_trials,searched_image[int(showy),int(showx)-np.abs(img.shape[1]-searched_image.shape[1]),i,:],kind='nearest',fill_value='extrapolate')
            dmtxinterp1[i,:] = finterp(np.linspace(np.min(DM_trials),np.max(DM_trials),100))
        dmtxinterp = np.zeros((100,100))
        for j in range(100):
            finterp = interp1d(widthtrials,dmtxinterp1[:,j],kind='nearest',fill_value='extrapolate')
            dmtxinterp[:,j] = finterp(np.linspace(np.min(widthtrials),np.max(widthtrials),100))
        ax.imshow(dmtxinterp.transpose(),origin="lower",extent=[min(widthtrials),max(widthtrials),min(DM_trials),max(DM_trials)],cmap='plasma',aspect='auto',vmin=0,vmax=np.nanmax(dmtxinterp))
        
        for i in range(len(widthtrials)):
            plt.axvline(widthtrials[i],color='grey',linewidth=1,linestyle='--',zorder=50)
        for i in range(len(DM_trials)):
            plt.axhline(DM_trials[i],color='grey',linewidth=1,linestyle='--',zorder=50)
        ax.axhline(DM_trials[dms][showidx],color='red',linestyle='--',linewidth=3,zorder=100)
        ax.axvline(widthtrials[wids][showidx],color='red',linestyle='--',linewidth=3,zorder=100)
        ax.set_xlim(np.min(widthtrials)-1,np.max(widthtrials)+1)
        ax.set_ylim(np.min(DM_trials)-1,np.max(DM_trials)+1)
        ax.set_xlabel("Width (Samples)")
        ax.set_ylabel(r"DM (pc/cc)")
        ax.set_facecolor('grey')
    elif searched_image is not None and imgdiff:
        #plot the DM transform thing for peak candidate
        showidx = np.nanargmax(snrs)
        printlog("SHOWING CAND " + str(showidx) + ", SNR=" + str(snrs[showidx]) + ", DM=" + str(DM_trials[dms][showidx]) + ", WID=" + str(widthtrials[wids][showidx]) + ", RAIDX=" + str(ras[showidx]) + ", DECIDX=" + str(decs[showidx]),output_file=output_file)
        showx,showy,showname = ras[showidx],decs[showidx],names[showidx]
        ax.set_title(showname)
        #plot snr vs width
        ax.step(widthtrials,searched_image[int(showy),int(showx)-np.abs(img.shape[1]-searched_image.shape[1]),:,0],where='post',linewidth=4)
        #ax.set_xscale("log")
        ax.set_xlim(np.min(widthtrials)-1,np.max(widthtrials)+1)
        ax.set_xlabel("Width (Samples)")
        ax.set_ylabel("S/N")

        ax.axvline(widthtrials[wids][showidx],color='red',linestyle='--',linewidth=3,zorder=100)
    else:
        ax.set_xlim(np.min(widthtrials)-1,np.max(widthtrials)+1)
        ax.set_ylim(np.min(DM_trials)-1,np.max(DM_trials)+1)
        ax.set_xlabel("Width (Samples)")
        ax.set_ylabel(r"DM (pc/cc)")
        ax.set_facecolor('grey')

    printlog("dm width plot done",output_file=output_file)

    printlog(timeseries,output_file=output_file)
    printlog(timeseries[0],output_file=output_file)
    printlog(tsamp_use*np.arange(len(timeseries[0]))/1000,output_file=output_file)
    printlog(names,output_file=output_file)
    
    #median subtracted timeseries
    ax = fig.add_subplot(gs[1,:])#ax=plt.subplot(3,2,3)
    for i in range(len(timeseries)):
        printlog("iter " + str(i),output_file=output_file)
        plt.step(tsamp_use*np.arange(len(timeseries[i]))/1000,timeseries[i] - np.nanmedian(timeseries[i]),alpha=1/(len(timeseries)),where='post',linewidth=4,label=names[i])
        #ax.legend(ncols=1 + int(len(timeseries)//5),loc="upper right",fontsize=20)
    ax.set_xlim(0,tsamp_use*img.shape[2]/1000)
    if timeseries_nondm:
        ax.set_title("Median Subtracted Timeseries")
    else:
        ax.set_title("De-dispersed Median Subtracted Timeseries")
    ax.set_ylim(ymin=0)
    printlog("timeseries plots done",output_file=output_file)

    if imgdiff:
        ax.set_xlabel("Time (s)")
    else:
        #show dynamic spectrum for highest S/N burst
        ax = fig.add_subplot(gs[2,:])#ax=plt.subplot(3,2,5)
        printlog("FROM PLOTTING, SNRS",output_file)
        printlog(snrs,output_file)
        printlog(names[np.argmax(snrs)],output_file)
        showx,showy,showname = ras[np.argmax(snrs)],decs[np.argmax(snrs)],names[np.argmax(snrs)]
        ax.set_title(showname)
        ax.imshow(img[int(showy),int(showx),:,:].transpose(),origin="upper",extent=[0,tsamp_use*img.shape[2]/1000,fmin,fmax],cmap='plasma',aspect='auto',vmin=0,vmax=np.nanmax(img[int(showy),int(showx),:,:].transpose()))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (MHz)")
        printlog("dynamic spectrum done",output_file=output_file)



    #show periodogram for highest S/N burst
    if plot_period:
        ax = fig.add_subplot(gs[3,1])
        printlog("FROM PLOTTING, PERIODOGRAM",output_file)
        showcand = np.argmax(snrs)
        timeseries = pcanddict[showcand]["timeseries"]
        initP = pcanddict[showcand]["initP_secs"]
        initPsamps = pcanddict[showcand]["initP_samps"]
        taxis = pcanddict[showcand]["taxis"]
        fineP = pcanddict[showcand]["fineP_secs"]
        resids = pcanddict[showcand]["resids"]
        trial_p_fine = pcanddict[showcand]["trial_p_cand_secs"]
        alphas = np.clip(timeseries/(np.nanmax(timeseries)/2 - np.nanmin(timeseries)),0.05,1)
        msizes = timeseries/(np.nanmax(timeseries)/2 - np.nanmin(timeseries))

        taxis = np.arange(len(timeseries))
        c=ax.imshow(resids,aspect='auto',extent=(0,1,np.nanmin(trial_p_fine),np.nanmax(trial_p_fine)),origin='lower',cmap='plasma')
        ax.axhline(fineP,color='red',linewidth=4)
        #.axvline(minresid[1]/trial_periods[peakidx[0]],color='red',linewidth=4)
        ax.set_title("Timing Residuals\nFine-timing Estimate: " + str(np.around(fineP,2)) + "s")
        plt.colorbar(mappable=c,ax=ax,label='Residuals (samples)')
        ax.set_xlabel("Phase")
        ax.set_ylabel("Trial Period (s)")
        printlog("DONE1",output_file)

        ax = fig.add_subplot(gs[3,0])
        ax.scatter((taxis%initPsamps)/initPsamps,(initP)*(taxis//initPsamps),alpha=alphas,s=msizes*1000)
        ax.set_ylabel("Time (s)")
        ax.set_xlabel("Phase")
        ax.set_title("Periodogram\nInitial Period Estimate: " + str(np.around(initP,2))+ "s")
        printlog("DONE2",output_file)






    t = "NSFRB " + isot
    if injection and not slow and not imgdiff: t = t + " (injection)"
    elif slow: t = t + " (slow)"
    elif imgdiff: t = t + " (slower)"

    #add parameters of peak candidate
    if imgdiff:
        t = t + "\n RA={a:.2f}, DEC={b:.2f}, W={d:.2f}s, SNR={g:.2f}".format(a=RA_axis[ras][np.argmax(snrs)],b=DEC_axis[decs][np.argmax(snrs)],d=widthtrials[wids][np.argmax(snrs)]*tsamp_use/1000,g=np.nanmax(snrs))
    else:
        t = t + "\n RA={a:.2f}, DEC={b:.2f}, DM={c:.2f}pc/cc, W={d:.2f}s, SNR={g:.2f}".format(a=RA_axis[ras][np.argmax(snrs)],b=DEC_axis[decs][np.argmax(snrs)],c=DM_trials[dms][np.argmax(snrs)],d=widthtrials[wids][np.argmax(snrs)]*tsamp_use/1000,g=np.nanmax(snrs))
    if 'predicts' in canddict.keys():
        if canddict['predicts'][np.argmax(snrs)]==0:
            t = t + "\nSource ({p:.2f})%".format(p=canddict['probs'][np.argmax(snrs)]*100)
        else:
            t = t + "\nRFI ({p:.2f})%".format(p=canddict['probs'][np.argmax(snrs)]*100)

    plt.suptitle(t)
    plt.savefig(output_dir + isot + "_NSFRBcandplot" + plotsuffix + ".png")
    if show:
        plt.show()
    else:
        plt.close()
    #plotting_now = False
    return isot + "_NSFRBcandplot" +plotsuffix + ".png"

def binary_plot(image_tesseract,SNRthresh,timestep_isot,RA_axis,DEC_axis,binary_file=binary_file):
    """
    This function writes a binary image representation to file
    for monitoring.
    """
    size = 10
    #take max over trial width, DM
    binplot = np.nanmax(image_tesseract,axis=(2,3))
    binplotdets = binplot >= SNRthresh



    #downscale
    print("BINPLOT: INITIAL AXIS SHAPE:",binplot.shape)
    gridsize_DEC,gridsize_RA = binplot.shape
    if len(RA_axis) != gridsize_RA:
        RA_axis = RA_axis[int((len(RA_axis)-gridsize_RA)//2):-((len(RA_axis)-gridsize_RA) - int((len(RA_axis)-gridsize_RA)//2))]
    if gridsize_DEC%size != 0 or gridsize_RA%size != 0:
        binplot = binplot[gridsize_DEC%size // 2: gridsize_DEC - (gridsize_DEC%size - (gridsize_DEC%size // 2)),
                        gridsize_RA%size // 2: gridsize_RA - (gridsize_RA%size - (gridsize_RA%size // 2))]
        RA_axis = RA_axis[gridsize_RA%size // 2: gridsize_RA - (gridsize_RA%size - (gridsize_RA%size // 2))]
        DEC_axis = DEC_axis[gridsize_DEC%size // 2: gridsize_DEC - (gridsize_DEC%size - (gridsize_DEC%size // 2))]
    print("BINPLOT: NEW AXIS SHAPE:",RA_axis.shape,DEC_axis.shape)
    gridsize_DEC,gridsize_RA = binplot.shape
    binplot = np.nanmax(binplot.reshape((size,gridsize_DEC//size,size,gridsize_RA//size)),(1,3))
    binplotdets = binplot >= SNRthresh
    RA_axis = RA_axis.reshape((size,gridsize_RA//size)).mean(1)
    DEC_axis = DEC_axis.reshape((size,gridsize_DEC//size)).mean(1)

    #normalize
    binplot = (binplot - np.nanpercentile(binplot,10))/(np.nanpercentile(binplot,90) - np.nanpercentile(binplot,10))
    binplot = np.array(10*np.clip(binplot,0,1),dtype=int)

    #write to file
    with open(binary_file,"w") as csvfile:
        csvfile.write(" "*(89//2 - len(timestep_isot)//2) + str(timestep_isot) + "\n")
        csvfile.write(" "*89 + "\t\tD\n")
        csvfile.write("-"*89 + "\n")
        wtr = csv.writer(csvfile,delimiter='\t')
        d = "DECLINATION"
        for i in range(binplot.shape[0])[::-1]:
            row = np.array(binplot[i,::-1],dtype=str)
            row[binplotdets[i,::-1]] = "X" #"◎"
            row[~binplotdets[i,::-1]] = "·"
            wtr.writerow(["|"] + list(row) + ["|"] + [str(np.around(DEC_axis[i],1))]  + [d[1 + i]])
            wtr.writerow([])
        csvfile.write("-"*89 + "\n")
        wtr.writerow(["|"] + list(np.array(np.around(RA_axis,1),dtype=str)[::-1]) + ["|"])
        csvfile.write(" "*(89//2 - 15//2) + "RIGHT ASCENSION" + "\n")
        
    csvfile.close()
    return








"""
Grafana doesn't like nsfrb, so I made my own 
"""
def timestatusplot(showsamps=30,update_time=T/1000,plotfile_searchtime=img_dir+"timestatusfile.txt"):
    """
    Pulls data from etcd in realtime and plots time for 
    imaging, sending, searching data in binary form
    """
    search_time_all = np.zeros(showsamps)
    search_txtime_all = np.zeros(showsamps)
    search_timouts_all = np.zeros(showsamps)
    interval=(T/1000/10)
    time_levels = np.arange(17)*interval
    packet_status_all = np.zeros(showsamps)
    packet_status_levels = np.arange(17)


    corr_rows = 2
    corr_cols = 8
    showsamps_corr = int(showsamps//4)
    image_time_all = [np.zeros(showsamps_corr)]*16
    tx_time_all = [np.zeros(showsamps_corr)]*16
    
    timelabel = "      "+ "".join(["  {:02.2f}  ".format((tsamp*i*9 + tsamp*5)/1000) for i in range(1 + int(showsamps//9))])
    tickmarks = "______"+ "".join(["____|____" for i in range(int(showsamps//9))])
    timelabel_corr = "      "+ "".join(["  {:02.2f}  ".format((tsamp*i*9 + tsamp*5)/1000) for i in range(int(showsamps_corr//9))]) + " "*6
    tickmarks_corr = "______"+ "".join(["____|____" for i in range(int(showsamps_corr//9))]) + "_"*4
    while True:
        
        ss=ETCD.get_dict("/mon/nsfrbsearchtiming")
        search_time_all = np.concatenate([search_time_all[1:],[ss["search_time"]]])
        search_txtime_all = np.concatenate([search_txtime_all[1:],[ss["search_tx_time"]]])
        #search_timouts_all = np.concatenate([search_timouts_all[1:],[ss["search_completed"]]])
        packet_status_all = np.concatenate([packet_status_all[1:],[ETCD.get_dict("/mon/nsfrbpackets")["dropped"]]])
        for i in range(16):
            dd = ETCD.get_dict("/mon/nsfrbtiming/"+str(i+1))
            image_time_all[i] = np.concatenate([image_time_all[i][1:],[dd["image_time"]]])
            tx_time_all[i] = np.concatenate([tx_time_all[i][1:],[dd["tx_time"]]])

        #quantize search time
        search_time_quantize = np.clip(np.array(search_time_all/interval,dtype=int),0,len(time_levels)-1)
        search_txtime_quantize = np.clip(np.array(search_txtime_all/interval,dtype=int),0,len(time_levels)-1)
        f = open(plotfile_searchtime,"w")
        f.write("Process Server Search Time (s)  " + " "*2*int(showsamps//3) + "Process Server Cand Stream Time (s)\n")
        for lev_i in np.arange(len(time_levels),dtype=int)[::-1]:
            p = np.array([" "]*len(search_time_all))
            p[search_time_quantize==lev_i] = "*"
            p2 = np.array([" "]*len(search_txtime_all))
            p2[search_txtime_quantize==lev_i] = "*"

            f.write(str("0" if time_levels[lev_i]<10 else "") + "{:.2f}".format(time_levels[lev_i]) + "|" + "".join(p) + "  " + str("0" if time_levels[lev_i]<10 else "") + "{:.2f}".format(time_levels[lev_i]) + "|" + "".join(p2) + "\n")
        f.write(tickmarks + "  " + tickmarks + "\n")
        f.write(timelabel + "  " + timelabel + "\n")
        f.write(" "*int(showsamps//3) + "Time Offset (s)  " + " "*int(showsamps//3) + " "*int(showsamps//3) + "Time Offset (s)\n")
        f.write("-"*showsamps*2 + "\n\n\n")




        #packet status
        f.write("Process Server Dropped Packets (0-16)\n")
        for lev_i in np.arange(len(packet_status_levels),dtype=int)[::-1]:
            p = np.array([" "]*len(packet_status_all))
            p[packet_status_all==lev_i] = "*"
            f.write(str(" " if packet_status_levels[lev_i]<10 else "") + str(packet_status_levels[lev_i]) + "   " +  "|" + "".join(p) + "\n")
        f.write(tickmarks + "\n")
        f.write(timelabel + "\n")
        f.write(" "*int(showsamps//3) + "Time Offset (s)\n")
        f.write("-"*showsamps + "\n\n\n")
        """
        #search timouts
        f.write("Process Server Search Completed (0=timeout, 1=success)\n")
        for lev_i in [1,0]:#np.arange(len(packet_status_levels),dtype=int)[::-1]:
            p = np.array([" "]*len(search_timouts_all))
            p[search_timouts_all==lev_i] = "*"
            f.write(str(" " if search_timouts_all[lev_i]<10 else "") + str(search_timouts_all[lev_i]) + "   " +  "|" + "".join(p) + "\n")
        f.write(tickmarks + "\n")
        f.write(timelabel + "\n")
        f.write(" "*int(showsamps//3) + "Time Offset (s)\n")
        f.write("-"*showsamps + "\n\n\n")
        """

        #quantize image time
        f.write("Corr Node Imaging Time (s)\n")
        for i in range(corr_rows):
            f.write("".join([f"sb{i*corr_cols+j}" + " "*(showsamps_corr+5) for j in range(corr_cols)]) + "\n")
            alllines = []
            for j in range(corr_cols):
                corridx = i*corr_cols + j
                image_time_quantize = np.clip(np.array(image_time_all[corridx]/interval,dtype=int),0,len(time_levels)-1)
                for lev_i in np.arange(len(time_levels),dtype=int)[::-1]:
                    p = np.array([" "]*len(image_time_all[corridx]))
                    p[image_time_quantize==lev_i] = "*"
                    if j == 0:
                        alllines.append(str("0" if time_levels[lev_i]<10 else "") + "{:.2f}".format(time_levels[lev_i]) + "|" + "".join(p))
                    else:
                        alllines[len(time_levels)-lev_i-1] += str("0" if time_levels[lev_i]<10 else "") + "{:.2f}".format(time_levels[lev_i]) + "|" + "".join(p)
            alllines.append("".join([tickmarks_corr]*corr_cols))
            alllines.append("".join([timelabel_corr]*corr_cols))
            alllines.append("".join([" "*int(showsamps_corr//3) + "Time Offset (s)      "]*corr_cols))
            for l in alllines:
                f.write(l + "\n")
            f.write("\n")
        f.write("-"*showsamps + "\n\n\n")

        #quantize tx time
        f.write("Corr Node TX Time (s)\n")
        for i in range(corr_rows):
            f.write("".join([f"sb{i*corr_cols+j}" + " "*(showsamps_corr+5) for j in range(corr_cols)]) + "\n")
            alllines = []
            for j in range(corr_cols):
                corridx = i*corr_cols + j
                tx_time_quantize = np.clip(np.array(tx_time_all[corridx]/interval,dtype=int),0,len(time_levels)-1)
                for lev_i in np.arange(len(time_levels),dtype=int)[::-1]:
                    p = np.array([" "]*len(tx_time_all[corridx]))
                    p[tx_time_quantize==lev_i] = "*"
                    if j == 0:
                        alllines.append(str("0" if time_levels[lev_i]<10 else "") + "{:.2f}".format(time_levels[lev_i]) + "|" + "".join(p))
                    else:
                        alllines[len(time_levels)-lev_i-1] += str("0" if time_levels[lev_i]<10 else "") + "{:.2f}".format(time_levels[lev_i]) + "|" + "".join(p)
            alllines.append("".join([tickmarks_corr]*corr_cols))
            alllines.append("".join([timelabel_corr]*corr_cols))
            alllines.append("".join([" "*int(showsamps_corr//3) + "Time Offset (s)      "]*corr_cols))
            for l in alllines:
                f.write(l + "\n")
            f.write("\n")


        f.close()




        #os.system("cat "+plotfile)
        time.sleep(update_time)
    return


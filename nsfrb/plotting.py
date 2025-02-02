import matplotlib
import matplotlib.animation as animation
from nsfrb import imaging
from nsfrb import pipeline
from dsamfs import utils as pu
from astropy.time import Time
from astropy import units as u
from nsfrb.planning import nvss_cat,atnf_cat,find_fast_vis_label
from nsfrb.config import tsamp,CH0,CH_WIDTH , AVERAGING_FACTOR,nsamps,NUM_CHANNELS
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
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,flagged_antennas,bad_antennas

"""
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")

binary_file = cwd + "-logfiles/binary_log.txt"
inject_file = cwd + "-injections/injections.csv"
"""
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


def search_plots_new(canddict,img,isot,RA_axis,DEC_axis,DM_trials,widthtrials,output_dir,show=True,vmax=1000,vmin=0,s100=100,injection=False,searched_image=None,timeseries=[],uv_diag=None,dec_obs=None):
    """
    Makes updated diagnostic plots for search system
    """
    gridsize = len(RA_axis)
    decs,ras,wids,dms=np.array(canddict['dec_idxs'],dtype=int),np.array(canddict['ra_idxs'],dtype=int),np.array(canddict['wid_idxs'],dtype=int),np.array(canddict['dm_idxs'],dtype=int)#np.unravel_index(np.arange(32*32*2*3)[(imgsearched>2500).flatten()],(32,32,3,2))#[1].shape
    snrs = np.array(canddict['snrs'])#imgsearched.flatten()[(imgsearched>2500).flatten()]
    names = np.array(canddict['names'])
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
    fig=plt.figure(figsize=(40,40))
    if injection:
        fig.patch.set_facecolor('red')
    gs = fig.add_gridspec(4,2)
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



    if uv_diag is not None and dec_obs is not None:
        ra_grid_2D_cut = ra_grid_2D[:,-searched_image.shape[1]:]
        dec_grid_2D_cut = dec_grid_2D[:,-searched_image.shape[1]:]
        if 'predicts' in canddict.keys():
            ra_grid_2D_cut = ra_grid_2D[:,-searched_image.shape[1]:]
            dec_grid_2D_cut = dec_grid_2D[:,-searched_image.shape[1]:]
            ax.scatter(ra_grid_2D_cut[decs[canddict['predicts']==0],ras[canddict['predicts']==0]].flatten(),
                    dec_grid_2D_cut[decs[canddict['predicts']==0],ras[canddict['predicts']==0]].flatten(),
                    c=snrs[canddict['predicts']==0],marker='o',cmap='jet',alpha=0.5,s=300*snrs[canddict['predicts']==0]/s100,vmin=vmin,vmax=vmax,linewidths=2,edgecolors='violet')
            ax.scatter(ra_grid_2D_cut[decs[canddict['predicts']==1],ras[canddict['predicts']==1]].flatten(),
                    dec_grid_2D_cut[decs[canddict['predicts']==1],ras[canddict['predicts']==1]].flatten(),
                    c=snrs[canddict['predicts']==1],marker='o',cmap='jet',alpha=0.5,s=300*snrs[canddict['predicts']==1]/s100,vmin=vmin,vmax=vmax,linewidths=2,edgecolors='limegreen')
        else:
            ax.scatter(ra_grid_2D_cut[decs,ras].flatten(),
                    dec_grid_2D_cut[decs,ras].flatten(),c=snrs,marker='o',cmap='jet',alpha=0.5,s=100*snrs/s100,vmin=vmin,vmax=vmax,linewidths=2,edgecolors='violet')
    else:
        if 'predicts' in canddict.keys():
            ax.scatter(RA_axis[ras][canddict['predicts']==0],DEC_axis[decs][canddict['predicts']==0],c=snrs[canddict['predicts']==0],marker='o',cmap='jet',alpha=0.5,s=300*snrs[canddict['predicts']==0]/s100,vmin=vmin,vmax=vmax,linewidths=2,edgecolors='violet')
            ax.scatter(RA_axis[ras][canddict['predicts']==1],DEC_axis[decs][canddict['predicts']==1],c=snrs[canddict['predicts']==1],marker='s',cmap='jet',alpha=0.5,s=300*snrs[canddict['predicts']==1]/s100,vmin=vmin,vmax=vmax,linewidths=2,edgecolors='violet')
        else:
            ax.scatter(RA_axis[ras],DEC_axis[decs],c=snrs,marker='o',cmap='jet',alpha=0.5,s=100*snrs/s100,vmin=vmin,vmax=vmax,linewidths=2,edgecolors='violet')#(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
    #nvss sources
    nvsspos,tmp,tmp = nvss_cat(Time(isot,format='isot').mjd,DEC_axis[len(DEC_axis)//2],sep=np.abs(np.max(DEC_axis)-np.min(DEC_axis))*u.deg)
    ax.plot(nvsspos.ra.value,nvsspos.dec.value,'o',markerfacecolor='none',markeredgecolor='blue',markersize=20,markeredgewidth=4,label='NVSS Source')
    #pulsars
    atnfpos,tmp = atnf_cat(Time(isot,format='isot').mjd,DEC_axis[len(DEC_axis)//2],sep=np.abs(np.max(DEC_axis)-np.min(DEC_axis))*u.deg)
    ax.plot(atnfpos.ra.value,atnfpos.dec.value,'s',markerfacecolor='none',markeredgecolor='blue',markersize=20,markeredgewidth=4,label='ATNF Pulsar')
    ax.legend(loc='lower right',frameon=True,facecolor='lightgrey')
    ax.axvline(RA_axis[gridsize//2],color='grey')
    ax.axhline(DEC_axis[gridsize//2],color='grey')
    ax.set_xlabel(r"RA ($^\circ$)")
    ax.set_ylabel(r"DEC ($^\circ$)")
    ax.invert_xaxis()
    ax.set_xlim(np.max(RA_axis),np.min(RA_axis))
    ax.set_ylim(np.min(DEC_axis),np.max(DEC_axis))

    ax = fig.add_subplot(gs[0,1])#ax=plt.subplot(3,2,2)
    if 'predicts' in canddict.keys():
        c=ax.scatter(widthtrials[wids][canddict['predicts']==0],
                DM_trials[dms][canddict['predicts']==0],c=snrs[canddict['predicts']==0],marker='o',cmap='jet',alpha=0.5,s=100*snrs[canddict['predicts']==0]/s100,vmin=vmin,vmax=vmax)#,alpha=(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
        c=ax.scatter(widthtrials[wids][canddict['predicts']==1],
                DM_trials[dms][canddict['predicts']==1],c=snrs[canddict['predicts']==1],marker='s',cmap='jet',alpha=0.5,s=100*snrs[canddict['predicts']==1]/s100,vmin=vmin,vmax=vmax)#,alpha=(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
    else:
        c=ax.scatter(widthtrials[wids],
                DM_trials[dms],c=snrs,marker='o',cmap='jet',alpha=0.5,s=100*snrs/s100,vmin=vmin,vmax=vmax)#,alpha=(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
    plt.colorbar(mappable=c,ax=ax,label='S/N')
    for i in widthtrials:
        ax.axvline(i,color='grey',linestyle='--')
    for i in DM_trials:
        ax.axhline(i,color='grey',linestyle='--')
    ax.set_xlim(0,np.max(widthtrials) + 1)
    ax.set_ylim(0,np.max(DM_trials) + 1)
    ax.set_xlabel("Width (Samples)")
    ax.set_ylabel(r"DM (pc/cc)")


    #timeseries
    ax = fig.add_subplot(gs[1,:])#ax=plt.subplot(3,2,3)
    for i in range(len(timeseries)):
        plt.step(tsamp*np.arange(len(timeseries[i]))/1000,timeseries[i],alpha=1/(0.5*len(timeseries)),where='post',linewidth=4,label=names[i])
    ax.legend(ncols=1 + int(len(timeseries)//5),loc="upper right",fontsize=20)
    ax.set_xlim(0,tsamp*img.shape[2]/1000)
    ax.set_title("De-dispersed Timeseries")

    #median subtracted timeseries
    ax = fig.add_subplot(gs[2,:])#ax=plt.subplot(3,2,3)
    for i in range(len(timeseries)):
        plt.step(tsamp*np.arange(len(timeseries[i]))/1000,timeseries[i] - np.nanmedian(timeseries[i]),alpha=1/(0.5*len(timeseries)),where='post',linewidth=4,label=names[i])
    #ax.legend(ncols=1 + int(len(timeseries)//5),loc="upper right",fontsize=20)
    ax.set_xlim(0,tsamp*img.shape[2]/1000)
    ax.set_title("De-dispersed Median Subtracted Timeseries")
    ax.set_ylim(ymin=0)

    #show dynamic spectrum for highest S/N burst
    ax = fig.add_subplot(gs[3,:])#ax=plt.subplot(3,2,5)
    showx,showy,showname = ras[np.argmax(snrs)],decs[np.argmax(snrs)],names[np.argmax(snrs)]
    ax.set_title(showname)
    ax.imshow(img[int(showy),int(showx)].transpose(),origin="lower",extent=[0,tsamp*img.shape[2]/1000,CH0,CH0 + CH_WIDTH * img.shape[3] * AVERAGING_FACTOR],cmap='plasma',aspect='auto')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")

    t = "NSFRB " + isot
    if injection: t = t + " (injection)"
    plt.suptitle(t)
    plt.savefig(output_dir + isot + "_NSFRBcandplot.png")
    if show:
        plt.show()
    else:
        plt.close()
    return isot + "_NSFRBcandplot.png"

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





#functions for imaging directly from files given candidate mjd
sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
def make_image_from_vis(T_interval,cand_mjd,full_array=True,image_size=1001,gif=False,visfile_dir=vis_dir,gulpsize=nsamps,nchan=2,headersize=16,binsize=5,bmin=20,sbimg=None,output_dir=vis_dir,viewsize=2):
    #visibility file
    fnum,offset = find_fast_vis_label(cand_mjd)
    print("file number: ",fnum)
    offset_gulp = int(offset//gulpsize)
    gulp_interval = int(T_interval//(tsamp*gulpsize))
    start_gulp = np.max([offset_gulp - gulp_interval,0])
    end_gulp = np.min([offset_gulp + gulp_interval,89])
    n_gulp = end_gulp-start_gulp
    print("gulps ",start_gulp,"-",end_gulp)
    all_imgs = []
    all_mjds = []
    for gulp in range(start_gulp,end_gulp):
        #read from file
        dat = None
        for i in range(len(corrs)):
            if sbimg is None or sbimg==i:
                try:
                    if visfile_dir==vis_dir:
                        dat_i,sb,mjd,dec = pipeline.read_raw_vis(visfile_dir + "/lxd110" + corrs[i] + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan,nsamps=gulpsize,gulp=gulp,headersize=headersize)
                    else:
                        dat_i,sb,mjd,dec = pipeline.read_raw_vis(visfile_dir + "/nsfrb_sb" + sbs[i] + "_" + str(fnum) + ".out",nchan=nchan,nsamps=gulpsize,gulp=gulp,headersize=headersize)
                    print(mjd,dec,sb)
                except Exception as exc:
                    print(exc)
                    dat_i = np.nan*np.ones((gulpsize, 4656, 2, 2))

                if dat is None:
                    dat = dat_i
                else:
                    dat = np.concatenate([dat,dat_i],axis=2)
        dat[np.isnan(dat)] = 0
    
        #flagging
        test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp_, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None,nsfrb=False)
        pt_dec = dec*np.pi/180.
        bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
        if full_array:
            dat, bname, blen, UVW, antenna_order = imaging.flag_vis(dat, bname, blen, UVW, antenna_order, bad_antennas, bmin=bmin)
        else:
            dat, bname, blen, UVW, antenna_order = imaging.flag_vis(dat, bname, blen, UVW, antenna_order, flagged_antennas, bmin=bmin)
        U = UVW[0,:,0]
        V = UVW[0,:,1]
        W = UVW[0,:,2]
        uv_diag=np.max(np.sqrt(U**2 + V**2))
        ff = 1.53-np.arange(8192)*0.25/8192
        fobs = ff[1024:1024+int(len(corrs)*NUM_CHANNELS/2)]
        fobs = np.reshape(fobs,(len(corrs)*2,int(NUM_CHANNELS/2/2))).mean(axis=1)

        #imaging
        for b in range(n_gulp*int(gulpsize//binsize)):
            dirty_img = np.zeros((image_size,image_size))
            for i in range(int(b*binsize),int((b+1)*binsize)):
                for j in range(dat.shape[2]):
                    for k in range(dat.shape[3]):
                        if ~np.all(np.isnan(dat[i:i+1,:,j,k])):
                            dirty_img += imaging.revised_robust_image(dat[i:i+1,:,j,k],
                                                       U/(2.998e8/fobs[j if sbimg is None else sbimg]/1e9),
                                                       V/(2.998e8/fobs[j if sbimg is None else sbimg]/1e9),
                                                       image_size,robust=2)
            all_imgs.append(dirty_img)
            all_mjds.append(mjd + (start_gulp*gulpsize*tsamp/1000/86400) + ((b+0.5)*binsize*tsamp/1000/86400))
        #make plot or gif
        dec_range = (dec-viewsize,dec+viewsize)
        ra_range = (imaging.get_ra(cand_mjd,dec)-viewsize,imaging.get_ra(cand_mjd,dec)+viewsize)
        std0=np.nanstd(all_imgs[0][:,:])
        if not gif:
            for i in range(len(all_imgs)):
                srcs,fs,tmp = nvss_cat(all_mjds[i],dec)
                psrs,names = atnf_cat(all_mjds[i],dec)
                ra_grid_2D,dec_grid_2D,elev = imaging.uv_to_pix(all_mjds[i],image_size,two_dim=True,manual=False,manual_RA_offset=0,output_file="",uv_diag=uv_diag,DEC=dec)
                
                plt.figure(figsize=(12,12))
                plt.scatter(ra_grid_2D[:,:].flatten(),dec_grid_2D[:,:].flatten(),c=(all_imgs[i]).flatten(),s=1,alpha=1)
                plt.scatter(srcs.ra.to(u.deg).value,srcs.dec.to(u.deg).value,marker='o',s=fs/30,facecolor='none',linewidth=1,c='red',alpha=0.5)
                plt.plot(psrs.ra.to(u.deg).value,psrs.dec.to(u.deg).value,'s',markersize=20,markerfacecolor='none',markeredgewidth=4,linewidth=4,markeredgecolor='red')
                plt.title(Time(all_mjds[i],format='mjd').isot)
                plt.xlim(ra_range[1],ra_range[0])
                plt.ylim(dec_range[0],dec_range[1])
                plt.xlabel(r"RA ($^\circ$)")
                plt.ylabel(r"DEC ($^\circ$)")
                plt.savefig(output_dir + (Time(all_mjds[i],format='mjd').isot + ("_{:02d}".format(sbimg) if sbimg is not None else "") + ".png"))
                plt.close()
   
        else:
            def update(i):
                srcs,fs,tmp = nvss_cat(all_mjds[i],dec)
                psrs,names = atnf_cat(all_mjds[i],dec)
                ra_grid_2D,dec_grid_2D,elev = imaging.uv_to_pix(all_mjds[i],image_size,two_dim=True,manual=False,manual_RA_offset=0,output_file="",uv_diag=uv_diag,DEC=dec)
                
                plt.gca()
                plt.scatter(ra_grid_2D[:,:].flatten(),dec_grid_2D[:,:].flatten(),c=(all_imgs[i]).flatten(),s=1,alpha=1)
                plt.scatter(srcs.ra.to(u.deg).value,srcs.dec.to(u.deg).value,marker='o',s=fs/30,facecolor='none',linewidth=1,c='red',alpha=0.5)
                plt.plot(psrs.ra.to(u.deg).value,psrs.dec.to(u.deg).value,'s',markersize=20,markerfacecolor='none',markeredgewidth=4,linewidth=4,markeredgecolor='red')
                plt.title(Time(all_mjds[i],format='mjd').isot)
                plt.xlim(ra_range[1],ra_range[0])
                plt.ylim(dec_range[0],dec_range[1])
                plt.xlabel(r"RA ($^\circ$)")
                plt.ylabel(r"DEC ($^\circ$)")
                return all_imgs[i],
            
            fig=plt.figure(figsize=(12,12))
            animation_fig = animation.FuncAnimation(fig,update,frames=len(all_imgs),interval=tsamp*binsize/1000)
            animation_fig.save(output_dir + (Time(cand_mjd,format='mjd').isot + ("_{:02d}".format(sbimg) if sbimg is not None else "") + ".gif"))
            plt.close()



        return all_imgs,all_mjds

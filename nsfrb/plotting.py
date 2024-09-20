import matplotlib
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
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")

binary_file = cwd + "-logfiles/binary_log.txt"
inject_file = cwd + "-injections/injections.csv"

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


def search_plots_new(canddict,img,isot,RA_axis,DEC_axis,DM_trials,widthtrials,output_dir,show=True,vmax=1000,vmin=0,s100=100,injection=False):
    """
    Makes updated diagnostic plots for search system
    """
    gridsize = len(RA_axis)
    decs,ras,wids,dms=np.array(canddict['dec_idxs'],dtype=int),np.array(canddict['ra_idxs'],dtype=int),np.array(canddict['wid_idxs'],dtype=int),np.array(canddict['dm_idxs'],dtype=int)#np.unravel_index(np.arange(32*32*2*3)[(imgsearched>2500).flatten()],(32,32,3,2))#[1].shape
    snrs = np.array(canddict['snrs'])#imgsearched.flatten()[(imgsearched>2500).flatten()]

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
    fig=plt.figure(figsize=(40,12))
    if injection:
        fig.patch.set_facecolor('red')
    ax = plt.subplot(1,2,1)

    ax.scatter(RA_axis[ras],DEC_axis[decs],c=snrs,marker='o',cmap='jet',alpha=0.5,s=100*snrs/s100,vmin=vmin,vmax=vmax,linewidths=2,edgecolors='violet')#(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
    #plt.contour(img.mean((2,3)),levels=3,colors='purple',linewidths=4)
    ax.imshow((img.mean((2,3)))[::-1,:],cmap='binary',aspect='auto',extent=[np.nanmin(RA_axis),np.nanmax(RA_axis),np.nanmin(DEC_axis),np.nanmax(DEC_axis)])
    
    ax.axvline(RA_axis[gridsize//2],color='grey')
    ax.axhline(DEC_axis[gridsize//2],color='grey')
    ax.set_xlabel(r"RA ($^\circ$)")
    ax.set_ylabel(r"DEC ($^\circ$)")
    ax.invert_xaxis()

    ax=plt.subplot(1,2,2)
    c=ax.scatter(widthtrials[wids],
                DM_trials[dms],c=snrs,marker='o',cmap='jet',alpha=0.5,s=100*snrs/s100,vmin=vmin,vmax=vmax)#,alpha=(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
    plt.colorbar(mappable=c,ax=ax,label='S/N')
    for i in widthtrials:
        ax.axvline(i,color='grey',linestyle='--')
    for i in DM_trials:
        ax.axhline(i,color='grey',linestyle='--')
    ax.set_xlim(0,np.max(widthtrials)*2)
    ax.set_ylim(0,np.max(DM_trials)*10)
    ax.set_xlabel("Width (Samples)")
    ax.set_ylabel(r"DM (pc/cc)")
    t = "NSFRB" + isot
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
    gridsize_DEC,gridsize_RA = binplot.shape
    if len(RA_axis) != gridsize_RA:
        RA_axis = RA_axis[int((len(RA_axis)-gridsize_RA)//2):-((len(RA_axis)-gridsize_RA) - int((len(RA_axis)-gridsize_RA)//2))]
    if gridsize_DEC%size != 0 or gridsize_RA%size != 0:
        binplot = binplot[gridsize_DEC%size // 2: gridsize_DEC - (gridsize_DEC%size - (gridsize_DEC%size // 2)),
                        gridsize_RA%size // 2: gridsize_RA - (gridsize_RA%size - (gridsize_RA%size // 2))]
        RA_axis = RA_axis[gridsize_RA%size // 2: gridsize_RA - (gridsize_RA%size - (gridsize_RA%size // 2))]
        DEC_axis = DEC_axis[gridsize_DEC%size // 2: gridsize_DEC - (gridsize_DEC%size - (gridsize_DEC%size // 2))]
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







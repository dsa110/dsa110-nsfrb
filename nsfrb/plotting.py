import matplotlib.pyplot as plt
import numpy as np

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


def search_plots_new(canddict,img,RA_axis,DEC_axis,DM_trials,widthtrials,output_dir,show=True,vmax=1000,vmin=0,s100=100):
    """
    Makes updated diagnostic plots for search system
    """
    gridsize = len(RA_axis)
    decs,ras,wids,dms=canddict['dec_idxs'],canddict['ra_idxs'],canddict['wid_idxs'],canddict['dm_idxs']#np.unravel_index(np.arange(32*32*2*3)[(imgsearched>2500).flatten()],(32,32,3,2))#[1].shape
    snrs = canddict['snrs']#imgsearched.flatten()[(imgsearched>2500).flatten()]



    plt.figure(figsize=(40,12))
    plt.subplot(1,2,1)
    plt.scatter(ras,decs,c=snrs,marker='o',cmap='jet',alpha=0.5,s=100*snrs/s100,vmin=vmin,vmax=vmax)#(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
    plt.contour(img.mean((2,3)),levels=3,colors='purple',linewidths=4)
    #plt.imshow(img.mean((2,3)),cmap='pink_r',aspect='auto')
    plt.axvline(gridsize//2,color='grey')
    plt.axhline(gridsize//2,color='grey')
    plt.xlabel("RA index")
    plt.ylabel("DEC index")

    plt.subplot(1,2,2)
    plt.scatter(widthtrials[wids],
                DM_trials[dms],c=snrs,marker='o',cmap='jet',alpha=0.5,s=100*snrs/s100,vmin=vmin,vmax=vmax)#,alpha=(snrs-np.nanmin(snrs))/(2*np.nanmax(snrs)-np.nanmin(snrs)))
    plt.colorbar(label='S/N')
    for i in widthtrials:
        plt.axvline(i,color='grey',linestyle='--')
    for i in DM_trials:
        plt.axhline(i,color='grey',linestyle='--')
    plt.xlim(0,np.max(widthtrials)*2)
    plt.ylim(0,np.max(DM_trials)*10)
    plt.xlabel("Width (Samples)")
    plt.ylabel("DM (pc/cc)")
    plt.savefig(output_dir + "diagnostic_RA_DEC.png")
    if show:
        plt.show()
    else:
        plt.close()
    return

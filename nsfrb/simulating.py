import numpy as np
from scipy.fftpack import ifftshift, ifft2
from scipy.stats import linregress
import time
import os
import pandas as pd
import random
from antpos.utils import get_itrf
from nsfrb.config import IMAGE_SIZE, c,fmin,fmax,tsamp    
import sys
from PIL import Image,ImageOps
from nsfrb import config
from scipy.stats import norm


def get_all_coordinates(flagged_antennas=[],return_names=False):
    """
    COPY OF get_core_coordinates without cutting out outriggers

    Extracts core antenna coordinates from a dataframe.

    Parameters:
    df: DataFrame containing antenna coordinates with columns 'x_m', 'y_m', 'z_m'.

    Returns:
    x_core, y_core, z_core: core coordinates of the antennas.
    """
    df = get_itrf()
    x_m = df['x_m'].values
    y_m = df['y_m'].values
    z_m = df['z_m'].values
    
    # Skip flagged antennas
    core_mask = np.array([list(df.index)[i] not in flagged_antennas for i in range(len(df))])

    print(list(df.index[core_mask]))

    # Extract core coordinates using the mask
    x_core = x_m[core_mask]
    y_core = y_m[core_mask]
    z_core = z_m[core_mask]


    #also return indices of which ones are in the core
    # Define core antenna limits
    #x_min, x_max = -2.41e6, -2.4092e6
    #y_min, y_max = -4.47830e6, -4.47775e6
    #core_idxs = np.arange(len(x_core),dtype=int)[(x_core > x_min) & (x_core < x_max) & (y_core > y_min) & (y_core < y_max)]
    if return_names: return x_core,y_core,z_core,list(df.index[core_mask])
    else: return x_core, y_core, z_core

def get_core_coordinates(flagged_antennas=[],return_names=False):
    """
    Extracts core antenna coordinates from a dataframe.

    Parameters:
    df: DataFrame containing antenna coordinates with columns 'x_m', 'y_m', 'z_m'.

    Returns:
    x_core, y_core, z_core: core coordinates of the antennas.
    """
    df = get_itrf()
    x_m = df['x_m'].values
    y_m = df['y_m'].values
    z_m = df['z_m'].values

    # Define core antenna limits
    x_min, x_max = -2.41e6, -2.4092e6
    y_min, y_max = -4.47830e6, -4.47775e6

    # Skip flagged antennas
    flag_mask = np.array([list(df.index)[i] not in flagged_antennas for i in range(len(df))])

    # Create a mask for core antennas
    core_mask = (x_m > x_min) & (x_m < x_max) & (y_m > y_min) & (y_m < y_max) & flag_mask
    print(list(df.index[core_mask]))

    # Extract core coordinates using the mask
    x_core = x_m[core_mask]
    y_core = y_m[core_mask]
    z_core = z_m[core_mask]

    if return_names: return x_core,y_core,z_core,list(df.index[core_mask])
    else: return x_core, y_core, z_core



def compute_uvw(x_m, y_m, z_m, HA, Dec):

    """
    Computes the u, v, w coordinates for baseline vectors between pairs of antennas.

    Parameters:
    x_m, y_m, z_m: Arrays of x, y, and z coordinates of the antennas in meters.
    HA: Hour Angle in radians.
    Dec: Declination in radians.

    Returns:
    Arrays of u, v, w coordinates for each baseline.
    """
    N = len(x_m)
    u, v, w = [], [], []

    for i in range(N):
        for j in range(i + 1, N):
            dx = x_m[j] - x_m[i]
            dy = y_m[j] - y_m[i]
            dz = z_m[j] - z_m[i]

            u_ij = dx * np.sin(HA) + dy * np.cos(HA)
            v_ij = -dx * np.sin(Dec) * np.cos(HA) + dy * np.sin(Dec) * np.sin(HA) + dz * np.cos(Dec)
            w_ij = dx * np.cos(Dec) * np.cos(HA) - dy * np.cos(Dec) * np.sin(HA) + dz * np.sin(Dec)

            u.append(u_ij)
            v.append(v_ij)
            w.append(w_ij)

    return np.array(u), np.array(v), np.array(w)


def apply_phase_shift(visibilities, u_core, v_core, u_shift_rad, v_shift_rad):
    """
    Apply a phase shift to the visibility data.

    Parameters:
    visibilities (array): Array of complex visibilities.
    u_core (array): Array of u-coordinates of the visibility points.
    v_core (array): Array of v-coordinates of the visibility points.
    u_shift_rad (float): Phase shift in the u-direction (radians).
    v_shift_rad (float): Phase shift in the v-direction (radians).

    Returns:
    array: The visibility data after applying the phase shift.
    """
    phase_shift = np.exp(1j * (u_shift_rad * u_core + v_shift_rad * v_core))
    return visibilities * phase_shift

def add_complex_gaussian_noise(V, mean=0, std_dev=1):
    """
    Add complex Gaussian noise to visibility data for better simulations.

    Parameters:
    V (array):visibility data.
    mean (float, optional): Mean of the Gaussian noise (default = 0).
    std_dev (float, optional): Std of the noise. (default = 1). For better simulation results should be increased.

    Returns:
    array: Vis data with noise.
    """
    real_noise = np.random.normal(mean, std_dev, V.shape)
    imag_noise = np.random.normal(mean, std_dev, V.shape)

    return V + real_noise + 1j*imag_noise

def apply_spectral_index(visibility, frequency, reference_frequency, spectral_index):
    """
    Apply a spectral index scaling to vis data.

    Parameters:
    visibility (array): vis data.
    frequency (float): Frequency at which the visibility data is observed.
    reference_frequency (float): Reference frequency for the spectral index.
    spectral_index (float): Spectral index to be applied.

    Returns:
    array: new vis data.
    """
    scaling_factor = (frequency / reference_frequency)**spectral_index
    return visibility * scaling_factor

def rfi_source_position(x_core, y_core, z_core, azimuth, elevation, distance=1e7):
    """
    Calculate the position of an RFI source based on azimuth and elevation.

    Parameters:
    x_core, y_core, z_core (arrays): Coordinates of core antennas.
    azimuth (float): Azimuth angle of the RFI source (in degrees).
    elevation (float): Elevation angle of the RFI source (in degrees).
    distance (float, optional): Distance to the RFI source. Default is 1e7 meters.

    Returns:
    tuple: Position coordinates (x, y, z) of the RFI source.
    """
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    x_mean = np.mean(x_core)
    y_mean = np.mean(y_core)
    z_mean = np.mean(z_core)

    # RFI source position based on azimuth and elevation
    rfi_x = x_mean + distance * np.cos(elevation) * np.sin(azimuth)
    rfi_y = y_mean + distance * np.cos(elevation) * np.cos(azimuth)
    rfi_z = z_mean + distance * np.sin(elevation)

    return rfi_x, rfi_y, rfi_z

def simulate_near_field_rfi(x, y, z, visibilities, rfi_position, rfi_amplitude, frequency):
    """
    Simulate the effect of near-field RFI on visibilities.

    Parameters:
    x,y,z (arrays): Antenna coordinates.
    visibilities (array): Visibility data to be affected.
    rfi_position (tuple): Coordinates of the RFI source.
    rfi_amplitude (float): Amplitude of the RFI signal.
    frequency (float): Frequency of observation.

    Returns:
    array: Modified visibilities with near-field RFI effects.
    """
    rfi_x, rfi_y, rfi_z = rfi_position

    deltas = np.sqrt((x - rfi_x) ** 2 + (y - rfi_y) ** 2 + (z - rfi_z) ** 2)
    delays = deltas / c

    rfi_signal = rfi_amplitude * np.exp(2j * np.pi * frequency * delays)

    num_antennas = len(x)
    k = 0
    for i in range(num_antennas):
        for j in range(i + 1, num_antennas):
            visibilities[k] += rfi_signal[i] + rfi_signal[j]
            k += 1

    return visibilities

def apply_phase_shift_to_rfi(rfi, u, v, u_shift_pixels, v_shift_pixels, pixel_resolution):
    """
    Applies a phase shift to the given data.

    Parameters:
    - rfi (ndarray): The RFI data to apply the phase shift to.
    - u (ndarray): The u-coordinates of the RFI data.
    - v (ndarray): The v-coordinates of the RFI data.
    - u_shift_pixels (float): The amount of shift in u-direction in pixels.
    - v_shift_pixels (float): The amount of shift in v-direction in pixels.
    - pixel_resolution (float): The resolution of each pixel.

    Returns:
    - ndarray: The RFI data after applying the phase shift.
    """
    u_shift = u_shift_pixels * pixel_resolution
    v_shift = v_shift_pixels * pixel_resolution

    u_shift_rad = 2 * np.pi * u * u_shift
    v_shift_rad = 2 * np.pi * v * v_shift

    shift_phase = np.exp(1j * (u_shift_rad + v_shift_rad))
    return rfi * shift_phase

def simulate_far_field_rfi(u, v, NUM_CHANNELS, num_baselines, pixel_resolution, noise_std_dev=5, desired_snr=7):
    """
    Simulates the far-field RFI for a given set of parameters.

    Args:
        u (array-like): The u-coordinates of the baselines.
        v (array-like): The v-coordinates of the baselines.
        NUM_CHANNELS (int): The number of frequency channels.
        num_baselines (int): The number of baselines.
        pixel_resolution (float): The pixel resolution of the image.
        noise_std_dev (float, optional): The standard deviation of the noise. Defaults to 5.#print(str(datagridsizecut) + " " + str(datagridsize) + " " + str(gridsize),file=fout) 
        desired_snr (float, optional): The desired signal-to-noise ratio (SNR) of the RFI. Defaults to 7.

    Returns:
        array-like: The simulated RFI with shape (NUM_CHANNELS, num_baselines).

    """
    rfi_amplitude = noise_std_dev * desired_snr
    rfi = np.zeros((NUM_CHANNELS, num_baselines), dtype=complex)

    num_affected_channels = np.random.randint(5, 15)  # Reasonable number of affected channels
    affected_freqs = np.random.choice(NUM_CHANNELS, size=num_affected_channels, replace=False)
    rfi[affected_freqs, :] = np.full((len(affected_freqs), num_baselines), rfi_amplitude)

    #u_shift_pixels = np.random.randint(-IMAGE_SIZE//2, IMAGE_SIZE//2)
    #v_shift_pixels = np.random.randint(-IMAGE_SIZE//2, IMAGE_SIZE//2)
    u_shift_pixels = 0
    v_shift_pixels = 0
    rfi = apply_phase_shift_to_rfi(rfi, u, v, u_shift_pixels, v_shift_pixels, pixel_resolution)

    return rfi



"""
Below are Myles's functions to quickly simulated injections and a model PSF based on Nikita's simualtions
"""
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
log_file = cwd + "-logfiles/inject_log.txt"
"""
def make_PSF_cube(gridsize=config.gridsize,nchans=config.nchans,nsamps=config.nsamps,RFI=False,output_file=log_file,datagridsize=256):
    
    #This function creates a frequency-dependent PSF based on Nikita's source simulation pipeline. It
    #uses pre-defined images and downsamples to the desired resolution. The PSF is duplicated along the time
    #axis.   
    
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    #get pngs for a point source from Nikita's images
    dirname = cwd + "/simulations_and_classifications/src_examples/observation_2/images/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/simulations_and_classifications/src_examples/observation_2/images/"
    pngs = os.listdir(dirname)
    sourceimg = np.zeros((gridsize,gridsize,nsamps,nchans))
    freqs = []
    fs = []

    print("Creating PSF with shape " + str(sourceimg.shape) + " using " + str(datagridsize) + "x" + str(datagridsize) + " images in " + dirname + "...",file=fout)
    for png in pngs:
        print(png,file=fout)
        if ".png" in png:
            #get frequency
            freq = float(png[png.index("avg_") + 4: png.index("avg_") + 11])
            freqs.append(freq)
            fs.append(png)


    #need to check that datagridsize/gridsize compatible
    if datagridsize > gridsize and datagridsize%gridsize != 0:
        diff = datagridsize%gridsize
        datagridsizecut = datagridsize - diff
    elif datagridsize > gridsize:
        diff = 0
        datagridsizecut = datagridsize


    #print(str(datagridsizecut) + " " + str(datagridsize) + " " + str(gridsize),file=fout) 
    if datagridsize > gridsize:
        print("Downsampling by factor " + str(datagridsizecut//gridsize) + "...",file=fout,end="")
    freqs_sorted = np.sort(freqs)
    fs_sorted = [x for x, _ in sorted(zip(fs, freqs))]
    #downsample and copy over time and frequency axes
    for i in range(nchans):
        for j in range(nsamps):

            #print(np.asarray(ImageOps.grayscale(Image.open(dirname + fs_sorted[i]))).shape)
            fullim = np.asarray(ImageOps.grayscale(Image.open(dirname + fs_sorted[i])))
            print(fullim.shape,file=fout)

            if datagridsize == gridsize:
                sourceimg[:,:,j,i] = fullim

            elif datagridsize < gridsize:
                diff = gridsize - datagridsize
                fullim = np.pad(fullim, (diff//2,diff - (diff//2)),mode='constant')
                sourceimg[:,:,j,i] = fullim

            elif datagridsize > gridsize:
                if datagridsize%gridsize != 0:
                    fullim = fullim[diff//2:(diff//2) + datagridsizecut,diff//2:(diff//2)+ datagridsizecut]

                sourceimg[:,:,j,i] = fullim.reshape((gridsize,datagridsize//gridsize,gridsize,datagridsize//gridsize)).mean((1,3))

    #roll if not perfectly centered
    maxpix = tuple(np.array(np.unravel_index(np.argmax(sourceimg[:,:,0,0].flatten()),(gridsize,gridsize))))
    centerpix = ((gridsize//2) - 1,(gridsize//2) - 1)
    if maxpix != centerpix:

        rolledPSFimg = np.roll(np.roll(sourceimg,shift=centerpix[0]-maxpix[0],axis=0),shift=centerpix[1]-maxpix[1],axis=1)
        rolledPSFimg[gridsize - (maxpix[0]-centerpix[0]):,:,:,:] = 0
        rolledPSFimg[:,gridsize - (maxpix[1]-centerpix[1]):,:,:] = 0
    else: rolledPSFimg = PSFimg
    #cutout image
    #PSFimg = rolledPSFimg[gridsize//2:gridsize//2 + gridsize,gridsize//2:gridsize//2 + gridsize]

    print("Complete!",file=fout)
    if output_file != "":
        fout.close()
    return rolledPSFimg



def make_image_cube(PSFimg,snr=1000,width=5,loc=0.5,gridsize=config.gridsize,nchans=config.nchans,nsamps=config.nsamps,RFI=False,DM=0,output_file=log_file,datagridsize=256):
    #get pngs
    #
    #This function makes test images with finite width using Nikita's test pngs
    #



    dirname = cwd + "/simulations_and_classifications/src_examples/observation_2/images/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/simulations_and_classifications/src_examples/observation_2/images/"#testimgs_2024-03-18/"#{a}x{a}_images/"#src_examples/observation_1/images/".format(a=gridsize)
    pngs = os.listdir(dirname)
    sourceimg = np.zeros((gridsize,gridsize,nsamps,nchans))
    freqs = []
    fs = []

    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #need to rescale by 2
    snr = snr/2

    for png in pngs:
        print(png,file=fout)
        if ".png" in png:
            #get frequency
            freq = float(png[png.index("avg_") + 4: png.index("avg_") + 11])
            freqs.append(freq)
            fs.append(png)

    #need to check that datagridsize/gridsize compatible
    if datagridsize > gridsize and datagridsize%gridsize != 0:
        diff = datagridsize%gridsize
        datagridsizecut = datagridsize - diff
    elif datagridsize > gridsize:
        diff = 0
        datagridsizecut = datagridsize


    #print(str(datagridsizecut) + " " + str(datagridsize) + " " + str(gridsize),file=fout) 
    if datagridsize > gridsize:
        print("Downsampling by factor " + str(datagridsizecut//gridsize) + "...",file=fout,end="")
    freqs_sorted = np.sort(freqs)
    fs_sorted = [x for x, _ in sorted(zip(fs, freqs))]
    #downsample and copy over time and frequency axes
    for i in range(nchans):
        for j in range(nsamps):

            #print(np.asarray(ImageOps.grayscale(Image.open(dirname + fs_sorted[i]))).shape)
            fullim = np.asarray(ImageOps.grayscale(Image.open(dirname + fs_sorted[i])))
            print(fullim.shape,file=fout)

            if datagridsize == gridsize:
                sourceimg[:,:,j,i] = fullim

            elif datagridsize < gridsize:
                diff = gridsize - datagridsize
                fullim = np.pad(fullim, (diff//2,diff - (diff//2)),mode='constant')
                sourceimg[:,:,j,i] = fullim

            elif datagridsize > gridsize:
                if datagridsize%gridsize != 0:
                    fullim = fullim[diff//2:(diff//2) + datagridsizecut,diff//2:(diff//2)+ datagridsizecut]

                sourceimg[:,:,j,i] = fullim.reshape((gridsize,datagridsize//gridsize,gridsize,datagridsize//gridsize)).mean((1,3))


    #now add noise based on the SNR
    #PSFimg = make_PSF_cube(loc=loc,gridsize=gridsize,nchans=nchans,nsamps=nsamps)
    #sourceimg = sourceimg[gridsize//2:gridsize//2 + gridsize,gridsize//2:gridsize//2 + gridsize]
    noises = []
    for i in range(nchans):
        sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i] = sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]/(np.sum((PSFimg*sourceimg)[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]))#/np.sum(PSFimg[:,:,:,i]))


        print(np.sum((PSFimg*sourceimg)[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]),np.sum(PSFimg[:,:,:,i]),file=fout)


        #img[16,16,500:500+wid,:] = snr/wid
        sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i] = sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]*snr#/np.sum(PSFimg[:,:,0,i])

        sourceimg[:,:,:int(loc*nsamps),:] = 0
        sourceimg[:,:,int(loc*nsamps) + width:,:] = 0

    #if DM is given, disperse before adding noise
    if DM != 0:
        sourceimg_dm = np.zeros(sourceimg.shape)
        freq_axis = np.linspace(fmin,fmax,nchans)
        for i in range(gridsize):
            for j in range(gridsize):
                tdelays = DM*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
                tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
                tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
                tdelays_frac = tdelays/tsamp - tdelays_idx_low

                for k in range(nchans):
                    #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac)
                    arrlow =  np.pad(sourceimg[i,j,:,k],((0,tdelays_idx_low[k])),mode="constant",constant_values=0)[tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                    arrhi =  np.pad(sourceimg[i,j,:,k],((0,tdelays_idx_hi[k])),mode="constant",constant_values=0)[tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

                    sourceimg_dm[i,j,:,k] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])

        
        #tmp,sourceimg = dedisperse_allDM(sourceimg,DM=-DM,keepfreqaxis=True)[:,:,:,:,0]
    else:
        sourceimg_dm = sourceimg
    for i in range(nchans):
        sourceimg_dm[:,:,:,i] += norm.rvs(loc=0,scale=np.sqrt(1/np.nansum(PSFimg[:,:,0,i])/width/nchans),size=(gridsize,gridsize,nsamps))
    #    noises.append(1/np.nansum(PSFimg[:,:,0,i])/width/nchans)

    if output_file != "":
        fout.close()
    return sourceimg_dm
"""

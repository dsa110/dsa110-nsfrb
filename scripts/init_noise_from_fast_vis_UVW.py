import argparse
from simulations_and_classifications import generate_PSF_images as scPSF
import h5py
from casatools import table
import numpy as np
from astropy.time import Time
import sys
import jax
import jax.numpy as jnp
"""
The purpose of this script is to initialize the noise from test fast visibility data on disk
"""


f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()

noise_dir = cwd + "-noise/"
sys.path.append(cwd+"/nsfrb/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/nsfrb/")
sys.path.append(cwd+"/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE, fmin,fmax,tsamp,nchans
from nsfrb.imaging import uniform_image, uv_to_pix
from nsfrb.TXclient import send_data  
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images  
from tqdm import tqdm 
import time
from scipy.stats import norm
import nsfrb.searching as sl
from nsfrb import jax_funcs
import os
vis_dir = os.environ["NSFRBDATA"] + "dsa110-nsfrb-fast-visibilities/"

def process_data_sb_VIS(fname, num_gulp, num_time_samples=25, verbose_flag=False, plot_uv_analysis_flag=False, plot_dirty_images_flag=False, num_channels=NUM_CHANNELS,averaging_factor=AVERAGING_FACTOR,image_size=IMAGE_SIZE,num_batches=1):
    """
    Process data from a CASA .ms table containing visibility data, which includes extracting the CORRECTED_DATA column data, 
    forming dirty images and sending them for further analysis.

    Args:
        num_gulp (int): Number of gulps to process. Total number of time samples = num_gulp * num_time_samples.
        num_time_samples (int, optional): Number of time samples per gulp. Defaults to 25.
        verbose_flag (bool, optional): Flag to enable verbose output. Defaults to False.
        plot_uv_analysis_flag (bool, optional): Flag to enable UV analysis plot. Defaults to False.
        plot_dirty_images_flag (bool, optional): Flag to enable dirty images plot. Defaults to False.

    Returns:
        None
    """
    if '.ms' in fname:
        tb = table()
        tb.open(fname)
        time_col = tb.getcol('TIME')  # Get the entire TIME column
    else:
        tb = h5py.File(fname)
        time_col = np.zeros(tb['Header']['time_array'].shape)
        tb['Header']['time_array'].read_direct(time_col)# Get the entire TIME column, saved as JULIAN DAY
        time_col = (time_col - 2400000.5)*86400 # convert to MJD in seconds, like ms files
        
   
    # Find the minimum and maximum times
    begin_time = np.min(time_col)
    end_time = np.max(time_col)

    if verbose_flag:
        begin_time_readable = Time(begin_time/86400, format='mjd').iso
        end_time_readable = Time(end_time/86400, format='mjd').iso

        print(f"Begin Time: {begin_time_readable}, End Time: {end_time_readable}")

    if  '.ms' in fname:
        if 'EXPOSURE' in tb.colnames():
            exposure_col = tb.getcol('EXPOSURE')
            # Assuming uniform exposure times
            tInt = exposure_col[0]
            print(f"Integration Time  from 'EXPOSURE' column: {tInt} seconds")
        elif 'INTERVAL' in tb.colnames():
            interval_col = tb.getcol('INTERVAL')
            # Assuming uniform intervals
            tInt = interval_col[0]
            print(f"Integration Time  from 'INTERVAL' column: {tInt} seconds")
        else:
            print("Neither 'EXPOSURE' nor 'INTERVAL' column found in the table.")
    else:
        tInt = tb['Header']['integration_time'][0]
    
    # Assert that begin_time + tInt * num_time_samples * num_gulp is less than or equal to end_time
    assert begin_time + tInt * num_time_samples * num_gulp <= end_time, "Invalid time range."

    all_vis_concat = np.zeros((num_time_samples*num_gulp,num_channels//averaging_factor),dtype=complex)
    for gulp in tqdm(range(num_gulp), desc="Processing gulps"):
        time_start = begin_time + tInt * num_time_samples * gulp
        time_end = begin_time + tInt * num_time_samples * (gulp + 1)

        if '.ms' in fname:
            # Use a table query to select rows within the num*time_samples time range
            selected_rows = tb.query(f'TIME >= {time_start} && TIME <= {time_end}')

            # Extract data columns from the selected_rows table
            data_selected = selected_rows.getcol('CORRECTED_DATA')  # 'CORRECTED_DATA' or 'MODEL_DATA' or 'DATA'
            uvw_selected = selected_rows.getcol('UVW')
            time_selected = selected_rows.getcol('TIME')
        
        else:
            idx_start = np.argmin(np.abs(time_col-time_start))
            idx_end = np.argmin(np.abs(time_col-time_end))
            data_selected = tb['Data']['visdata'][idx_start:idx_end,0,:,:].transpose((2,1,0))
            uvw_selected = tb['Header']['uvw_array'][idx_start:idx_end,:].transpose()
            time_selected = tb['Header']['time_array'][idx_start:idx_end]

        # Assuming time_selected is your array of time values
        time_diffs = np.diff(time_selected)  

        # Find indices where the difference is not zero (i.e., the time changes)
        time_change_indices = np.where(time_diffs != 0.0)[0] + 1  # Add 1 because np.diff reduces the array size by 1

        all_indices = np.append(time_change_indices, len(time_selected))
        print(len(all_indices),num_time_samples)
        assert len(all_indices) == num_time_samples, "Number of time samples does not match the expected value"

        start_idx = 0
        # Iterate over all blocks
        dirty_images_all = []
        k = 0
        for end_idx in all_indices:
            # Extract the current block for each array
            data_block = data_selected[:, :, start_idx:end_idx]
            uvw_block = uvw_selected[:, start_idx:end_idx]
            time_block = time_selected[start_idx:end_idx]
            vis_averaged = np.mean(data_block, axis=0)
            u = uvw_block[0, :]
            v = uvw_block[1, :]
            
            if plot_uv_analysis_flag:
                amplitude = np.abs(vis_averaged)
                phase = np.angle(vis_averaged)
                # Average amplitude and phase over frequency for the current block
                average_amplitude = np.mean(amplitude, axis=0)
                average_phase = np.mean(phase, axis=0)
                plot_uv_analysis(u, v, average_amplitude, average_phase, save_to_pdf=False, pdf_filename=f"{time_block[0]}_uv.pdf")

            dirty_images = []
            for i in range(0, num_channels, averaging_factor):
                chunk_V = vis_averaged[i:i + averaging_factor]
                all_vis_concat[(gulp*num_time_samples) + k,i] = np.nanmean(chunk_V)
                
                
                
            # Prepare for the next block
            start_idx = end_idx
            k += 1

        if '.ms' in fname:
            selected_rows.close()

    tb.close()
    return all_vis_concat

def get_noise_sb(dirty_images_all,num_gulp, num_time_samples=25, verbose_flag=False, plot_uv_analysis_flag=False, plot_dirty_images_flag=False, num_channels=NUM_CHANNELS,averaging_factor=AVERAGING_FACTOR,image_size=IMAGE_SIZE,num_batches=1):


    #for the injections, save the mean RMS in each channel
    raw_noise = np.mean(np.std(dirty_images_all,axis=2),axis=(0,1))
    np.save(noise_dir + "raw_noise_" + str(dirty_images_all.shape[0]) + "x" + str(dirty_images_all.shape[0]) + ".npy",raw_noise)

    for gulp in tqdm(range(num_gulp), desc="Estimating noise"):
        #esimate the noise by running the search pipeline on larger number of samples, only estimate noise up to -maxshift
        image_tesseract_filtered = dirty_images_all[:,:,gulp*num_time_samples:(gulp+1)*num_time_samples,:]
        tDM_max = (4.15)*sl.maxDM*((1/fmin/1e-3)**2 - (1/fmax/1e-3)**2) #ms
        maxshift = int(np.ceil(tDM_max/tsamp))
        truensamps = num_time_samples - maxshift
        nsamps = num_time_samples
        freq_axis = np.linspace(fmin,fmax,nchans)

        print("here",nsamps,truensamps)
        corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append = sl.gen_dm_shifts(sl.DM_trials,freq_axis,tsamp,nsamps)

        PSF = scPSF.generate_PSF_images(sl.psf_dir,0,image_tesseract_filtered.shape[0]//2,True,truensamps) 
        
        full_boxcar_filter = sl.gen_boxcar_filter(sl.widthtrials,truensamps)

        prev_noise_N = 0
        prev_noise = np.zeros((len(sl.widthtrials),len(sl.DM_trials)))
        noiseth = 0.1

        jaxdev = 0
        allnoise = np.zeros((len(sl.widthtrials),len(sl.DM_trials)))

        print("Image: ",image_tesseract_filtered.shape)
        print("DM COrr Shifts:",corr_shifts_all_append.shape)
        print("time delays:",tdelays_frac_append.shape)
        print("boxcar:",full_boxcar_filter.shape)
        #matched filter
        image_tesseract_filtered = jax_funcs.matched_filter_fft_jit(jax.device_put(np.array(image_tesseract_filtered,dtype=np.float32),jax.devices()[0]),jax.device_put(np.array(PSF[:,:,0:1,:],dtype=np.float32),jax.devices()[0]))

        #print(image_tesseract_filtered)
        image_tesseract_searched = np.zeros((image_tesseract_filtered.shape[0],image_tesseract_filtered.shape[1],len(sl.widthtrials),len(sl.DM_trials)))
        batchsize = image_tesseract_filtered.shape[0]//num_batches
        for i in range(num_batches):
            for j in range(num_batches):
                outtup = jax_funcs.matched_filter_dedisp_snr_fft_jit_init(jax.device_put(np.array(image_tesseract_filtered[i*batchsize:(i+1)*batchsize,j*batchsize:(j+1)*batchsize,:,:],dtype=np.float32),jax.devices()[jaxdev]),
                                                                #jax.device_put(np.ones(PSF[i*batchsize:(i+1)*batchsize,j*batchsize:(j+1)*batchsize,0:1,:].shape,dtype=np.float32),jax.devices()[jaxdev]),
                                                                 jax.device_put(corr_shifts_all_append,jax.devices()[jaxdev]),
                                                                 jax.device_put(tdelays_frac_append,jax.devices()[jaxdev]),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[jaxdev]),
                                                                 jax.device_put(np.array(prev_noise,dtype=np.float16),jax.devices()[jaxdev]),
                                                                 prev_noise_N,noiseth)
                #print(outtup[1]) 
                allnoise += outtup[1]        
                image_tesseract_searched[i*batchsize:(i+1)*batchsize,j*batchsize:(j+1)*batchsize,:,:] = outtup[0]
        total_noise = allnoise/(num_batches*num_batches)
        writeonly = (gulp == 0)
        current_noise = (sl.noise_update_all(total_noise,image_tesseract_filtered.shape[0],image_tesseract_filtered.shape[1],sl.DM_trials,sl.widthtrials,writeonly=writeonly),1)
        print(current_noise) 
    return
        

def main():
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities form the .ms file.')
    parser.add_argument('fname',type=str,help='Path to file with 16 visibility filenames')
    parser.add_argument('num_gulp', type=int, help='Number of gulps')
    parser.add_argument('num_time_samples', type=int, default=25, help='Number of time samples to extract from the .ms file.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--plot-uv-analysis', action='store_true',default=False, help='Enable UV analysis plotting')
    parser.add_argument('--plot-dirty-images', action='store_true', default=False, help='Enable dirty images plotting')
    parser.add_argument('--num_batches',type=int,help='Number of batches',default=1)
    args = parser.parse_args()

    f = open(args.fname,"r")
    fnames = f.read().split("\n")[:-1]
    print(fnames)
    assert(len(fnames)==16)

    dirty_images_all_concat = np.zeros((IMAGE_SIZE,IMAGE_SIZE,args.num_time_samples*args.num_gulp,NUM_CHANNELS//AVERAGING_FACTOR))
    vis_all_channels = np.zeros((args.num_time_samples*args.num_gulp,NUM_CHANNELS//AVERAGING_FACTOR),dtype=complex)
    for i in range(NUM_CHANNELS//AVERAGING_FACTOR):
        vis_all_channels[:,i:i+1] = process_data_sb_VIS(vis_dir +"test_files/"+ fnames[i],args.num_gulp, args.num_time_samples, args.verbose, args.plot_uv_analysis, args.plot_dirty_images,num_channels=2,averaging_factor=2,num_batches=args.num_batches)
        print(i,"real:",np.nanmean(np.real(vis_all_channels[:,i:i+1])),np.nanstd(np.real(vis_all_channels[:,i:i+1])))
        print(i,"imag:",np.nanmean(np.imag(vis_all_channels[:,i:i+1])),np.nanstd(np.imag(vis_all_channels[:,i:i+1])))
        print("")
        #dirty_images_all_concat[:,:,:,i:i+1] = process_data_sb(fnames[i],args.num_gulp, args.num_time_samples, args.verbose, args.plot_uv_analysis, args.plot_dirty_images,num_channels=2,averaging_factor=2,num_batches=args.num_batches)
    np.save(noise_dir + "raw_vis_noise_real.npy",np.nanstd(np.real(vis_all_channels),axis=0))
    np.save(noise_dir + "raw_vis_noise_imag.npy",np.nanstd(np.imag(vis_all_channels),axis=0))
    #get_noise_sb(dirty_images_all_concat,args.num_gulp, args.num_time_samples, args.verbose, args.plot_uv_analysis, args.plot_dirty_images,num_channels=2,averaging_factor=2,num_batches=args.num_batches)

if __name__ == '__main__':
    main()

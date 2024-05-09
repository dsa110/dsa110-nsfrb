import argparse
from casatools import table
import numpy as np
from astropy.time import Time
import sys

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()

sys.path.append(cwd+"/nsfrb/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/nsfrb/")
sys.path.append(cwd+"/")#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb.config import NUM_CHANNELS, AVERAGING_FACTOR, IMAGE_SIZE
from nsfrb.imaging import uniform_image
from nsfrb.TXclient import send_data  
from nsfrb.plotting import plot_uv_analysis, plot_dirty_images  
from tqdm import tqdm 
import time
from scipy.stats import norm

def process_data(num_gulp, num_time_samples=25, verbose_flag=False, plot_uv_analysis_flag=False, plot_dirty_images_flag=False):
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
    tb = table()
    tb.open('/media/ubuntu/ssd/sherman/code/CORR20BACKUP/ubuntu/nkosogor/2023-10-03_1459+716.ms')#'/home/ubuntu/nkosogor/2023-10-03_1459+716.ms')

    time_col = tb.getcol('TIME')  # Get the entire TIME column
    # TIME is in Modified Julian Date (MJD) in seconds
    # Find the minimum and maximum times
    begin_time = np.min(time_col)
    end_time = np.max(time_col)

    if verbose_flag:
        begin_time_readable = Time(begin_time/86400, format='mjd').iso
        end_time_readable = Time(end_time/86400, format='mjd').iso

        print(f"Begin Time: {begin_time_readable}, End Time: {end_time_readable}")

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

    # Assert that begin_time + tInt * num_time_samples * num_gulp is less than or equal to end_time
    assert begin_time + tInt * num_time_samples * num_gulp <= end_time, "Invalid time range."

    for gulp in tqdm(range(num_gulp), desc="Processing gulps"):
        time_start = begin_time + tInt * num_time_samples * gulp
        time_end = begin_time + tInt * num_time_samples * (gulp + 1)

        # Use a table query to select rows within the num*time_samples time range
        selected_rows = tb.query(f'TIME >= {time_start} && TIME <= {time_end}')

        # Extract data columns from the selected_rows table
        data_selected = selected_rows.getcol('CORRECTED_DATA')  # 'CORRECTED_DATA' or 'MODEL_DATA' or 'DATA'
        uvw_selected = selected_rows.getcol('UVW')
        time_selected = selected_rows.getcol('TIME')

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
            for i in range(0, NUM_CHANNELS, AVERAGING_FACTOR):
                chunk_V = vis_averaged[i:i + AVERAGING_FACTOR]
                dirty_img = uniform_image(chunk_V, u, v, IMAGE_SIZE)
                dirty_images.append(dirty_img)

            if plot_dirty_images_flag:
                plot_dirty_images(dirty_images, save_to_pdf=False, pdf_filename=f"{time_block[0]}_dirty_images.pdf")

            dirty_images_all.append(dirty_images)
            # Prepare for the next block
            start_idx = end_idx

        time_start_isot = Time(time_start / 86400, format='mjd').isot

        # Send the dirty images to the TX client
        dirty_images_all = np.array(dirty_images_all)   
        # transposing to have the following shape (num_pix, num_pix, num_time_samples, num_channels)
        # Sending one sub-band at a time
       
        #pad with zeros and noise to test pipeline
        dirty_images_all = dirty_images_all.transpose((2, 3, 0, 1))
        dirty_images_all = np.pad(dirty_images_all,((0,0),(0,0),(11,12),(0,0)))
        print(dirty_images_all.shape)
        dirty_images_all += norm.rvs(loc=0,scale=np.nanmax(dirty_images_all)/2,size=dirty_images_all.shape)
 
        gridsize = 300
        for i in range(NUM_CHANNELS//AVERAGING_FACTOR):
            #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
            msg=send_data(time_start_isot, dirty_images_all[:,:,:,i] ,verbose=verbose_flag,retries=5,keepalive_time=10)
            if verbose_flag: print(msg)
            time.sleep(1)

        selected_rows.close()

    tb.close()

def main():
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities form the .ms file.')
    parser.add_argument('num_gulp', type=int, help='Number of gulps')
    parser.add_argument('num_time_samples', type=int, default=25, help='Number of time samples to extract from the .ms file.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--plot-uv-analysis', action='store_true',default=False, help='Enable UV analysis plotting')
    parser.add_argument('--plot-dirty-images', action='store_true', default=False, help='Enable dirty images plotting')
    args = parser.parse_args()

    process_data(args.num_time_samples, args.num_gulp, args.verbose, args.plot_uv_analysis, args.plot_dirty_images)

if __name__ == '__main__':
    main()

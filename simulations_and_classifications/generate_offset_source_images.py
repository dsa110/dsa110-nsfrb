import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  
from nsfrb.simulating import compute_uvw, add_complex_gaussian_noise, get_core_coordinates, apply_spectral_index, apply_phase_shift
from nsfrb.imaging import uniform_image
from nsfrb.config import NUM_CHANNELS, CH0, CH_WIDTH, AVERAGING_FACTOR, IMAGE_SIZE, c


def generate_offset_src_images(dataset_dir, num_observations, noise_std_low, noise_std_high, exclude_antenna_percentage, HA_point, HA_source, Dec_point, Dec_source, spectral_index_low, spectral_index_high, zoom_pix, tonumpy,inflate=False,noise_only=False):
    """
    This function generates images of sources observed with DSA-110 core antennas.
    It takes various parameters such as the dataset directory, the number of observations, 
    the bounds for noise standard deviation, excluded antenna percentage, HA, Dec, spectral index, and zoom level.
    It generates multiple observations with different random parameters and saves the images
    and metadata in the specified dataset directory.

    Parameters:
    - dataset_dir (str): Dataset directory.
    - num_observations (int): Number of generated observations.
    - noise_std_low (float): Lower bound for noise standard deviation.
    - noise_std_high (float): Upper bound for noise standard deviation.
    - exclude_antenna_percentage (list[float]): Lower and upper bounds for excluded antenna percentage.
    - HA_point (float): HA for center of image (Hour Angle).
    - HA_source (float): HA of source (Hour Angle).
    - Dec_point (float): DEC for center of image (Declination).
    - Dec_source (float): DEC of source (Declination).
    - spectral_index_low (float): Lower bound for spectral index.
    - spectral_index_high (float): Upper bound for spectral index.
    - zoom_pix (int): Number of pixels to zoom in.
    - tonumpy (bool): If set, save to .npy file
    - inflate (bool): If set and zoom_pix > IMAGE_SIZE//2, generates image of size 2*zoom_pix x 2*zoom_pix
    - noise_only (bool): If set, only generates noise in visibilities and images

    Returns:
    - None

    """
    dataset_dir = os.path.join(dataset_dir, 'src_examples')

    x_core, y_core, z_core = get_core_coordinates()
    #pixel_resolution = (0.20 / np.max(np.sqrt(x_core**2 + y_core**2))) / 3

    ANTENNA_COUNT = len(x_core)  # Assuming x_core length represents the antenna count
    EXCLUDE_ANTENNA_PERCENTAGE = np.random.uniform(exclude_antenna_percentage[0], exclude_antenna_percentage[1]) #number of excluded antennas

    for obs in tqdm(range(num_observations), desc="Progress"):
        desired_shift_pixels_x = 0
        desired_shift_pixels_y = 0
        ##HA = np.random.uniform(low=HA_low, high=HA_high)
        ##Dec = np.random.uniform(low=Dec_low, high=Dec_high)

        

        # Calculate u_shift and v_shift in wavelengths
        #u_shift = desired_shift_pixels_x * pixel_resolution
        #v_shift = desired_shift_pixels_y * pixel_resolution

        # Convert the shift values to radians
        #u_shift_rad = 2 * np.pi * u_shift
        #v_shift_rad = 2 * np.pi * v_shift

        exclude_antennas = np.random.choice(range(ANTENNA_COUNT), size=int(ANTENNA_COUNT * EXCLUDE_ANTENNA_PERCENTAGE), replace=False)

        u_core, v_core, w_core = compute_uvw(x_core, y_core, z_core, HA_point, Dec_point) #core antennas -- this will make the PSF show up at the center
        u_core_s, v_core_s, w_core_s = compute_uvw(x_core, y_core, z_core, HA_source, Dec_source) #use this to offset to the source

        V = [np.ones(len(u_core)) + 1j*np.ones(len(v_core)) for _ in range(NUM_CHANNELS)]

        u_core_new, v_core_new, w_core_new, V_new = [], [], [], []
        u_core_s_new, v_core_s_new, w_core_s_new = [], [], []

        for i in range(ANTENNA_COUNT):
            for j in range(i+1, ANTENNA_COUNT):
                if i not in exclude_antennas and j not in exclude_antennas:
                    index = ANTENNA_COUNT * i - (i * (i+1)) // 2 + j - i - 1
                    u_core_new.append(u_core[index])
                    v_core_new.append(v_core[index])
                    w_core_new.append(w_core[index])
                    
                    u_core_s_new.append(u_core_s[index])
                    v_core_s_new.append(v_core_s[index])
                    w_core_s_new.append(w_core_s[index])
                    V_new.append([v[index] for v in V])

        # Convert lists back to numpy arrays or appropriate data structures
        u_core, v_core, w_core = np.array(u_core_new), np.array(v_core_new), np.array(w_core_new)
        u_core_s, v_core_s, w_core_s = np.array(u_core_s_new), np.array(v_core_s_new), np.array(w_core_s_new)
        V = np.array(V_new).T 

        reference_frequency_MHz = np.random.uniform(CH0, CH0 + NUM_CHANNELS * CH_WIDTH)

        spectral_index = np.random.uniform(spectral_index_low, spectral_index_high) #random sp index from -2 to 2
        for i in range(NUM_CHANNELS):
            frequency_MHz = CH0 + i * CH_WIDTH  # Calculating the frequency for each channel
            V[i] = apply_spectral_index(V[i], frequency_MHz, reference_frequency_MHz, spectral_index)

        noise = np.random.uniform(noise_std_low, noise_std_high)
        if noise_only:
            V_noisy = [add_complex_gaussian_noise(np.zeros_like(v), std_dev=noise) for v in V]
        else:
            V_noisy = [add_complex_gaussian_noise(v, std_dev=noise) for v in V]

        dirty_images = []
        for i in range(0, NUM_CHANNELS, AVERAGING_FACTOR):
            chunk_V = V_noisy[i:i+AVERAGING_FACTOR]
            avg_freq = CH0 + CH_WIDTH * i + AVERAGING_FACTOR/2 * CH_WIDTH

            wavelength = c / (avg_freq * 1e6)
            chunk_u_core = u_core / wavelength
            chunk_v_core = v_core / wavelength
            chunk_u_s_core = u_core_s / wavelength
            chunk_v_s_core = v_core_s / wavelength

            # Apply the phase shift to chunk_V
            print("PHASE SHIFT APPLIED:",HA_source-HA_point,Dec_source-Dec_point)
            chunk_V = [apply_phase_shift(v, chunk_u_core, chunk_v_core, HA_source-HA_point, Dec_source-Dec_point) for v in chunk_V]
            
            
            #chunk_V_shifted = [apply_phase_shift(v, u_shift_rad, v_shift_rad) for v in chunk_V]
            if inflate:
                dirty_img = uniform_image(chunk_V, chunk_u_core, chunk_v_core, zoom_pix*2)
            else:
                dirty_img = uniform_image(chunk_V, chunk_u_core, chunk_v_core, IMAGE_SIZE)
            dirty_images.append(dirty_img)

        # Creating necessary directories
        observation_dir = os.path.join(dataset_dir, f'observation_{obs+1}/images')
        os.makedirs(observation_dir, exist_ok=True)

        # Metadata
        metadata = []

        # Saving the images and collecting metadata
        if tonumpy:
            dirty_img_all = np.zeros((zoom_pix*2,zoom_pix*2,int(NUM_CHANNELS//AVERAGING_FACTOR)))
        for i, dirty_img in enumerate(dirty_images):
            avg_freq = CH0 + CH_WIDTH * i * AVERAGING_FACTOR
            filename = f'subband_avg_{avg_freq:.2f}_MHz.png'
            filepath = os.path.join(observation_dir, filename)
            im_zoom = np.array(np.fliplr(np.abs(dirty_img.T)))
            if not inflate:
                im_zoom = im_zoom[(IMAGE_SIZE // 2 - zoom_pix):(IMAGE_SIZE // 2 + zoom_pix), (IMAGE_SIZE // 2 - zoom_pix):(IMAGE_SIZE // 2 + zoom_pix)]
            plt.imsave(filepath, im_zoom, cmap='gray')
            if tonumpy:
                dirty_img_all[:,:,i] = im_zoom
        if tonumpy:
            np.save(os.path.join(observation_dir,f'final_img_{HA_point:.2f}_hr_{Dec_point:.2f}_deg.npy'),dirty_img_all)
        metadata.append({
            #'filename': filename,
            'desired_shift_pixels_x': desired_shift_pixels_x,
            'desired_shift_pixels_y': desired_shift_pixels_y,
            'HA': HA_point,
            'Dec': Dec_point,
            'Noise': noise,
            'Spectral index': spectral_index,
        })

        # Saving metadata to a CSV file
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(dataset_dir, f'observation_{obs+1}/metadata.csv'), index=False)

    print("All observations saved.")
    if tonumpy:
        return dirty_img_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images of sources observed with DSA-110 core antennas.')
    parser.add_argument('--dataset_dir', type=str, default='/Users/nikita/dsa110-nsfrb/simulations_and_classifications', help='Dataset directory')
    parser.add_argument('--num_observations', type=int, default=5, help='Number of generated observations')
    parser.add_argument('--noise_std_low', type=float, default=1, help='Lower bound for noise standard deviation')
    parser.add_argument('--noise_std_high', type=float, default=2, help='Upper bound for noise standard deviation')
    parser.add_argument('--exclude_antenna_percentage', type=float, nargs=2, default=[0, 0.15], help='Lower and upper bounds for excluded antenna percentage')
    parser.add_argument('--HA_point', type=float, default=0, help='Lower bound for HA')
    parser.add_argument('--HA_source', type=float, default=0, help='Upper bound for HA')
    parser.add_argument('--Dec_point', type=float, default=0, help='Lower bound for Dec')
    parser.add_argument('--Dec_source', type=float, default=0, help='Upper bound for Dec')
    parser.add_argument('--spectral_index_low', type=float, default=-2, help='Lower bound for spectral index')
    parser.add_argument('--spectral_index_high', type=float, default=2, help='Upper bound for spectral index')
    parser.add_argument('--zoom_pix', type=int, default=25, help='Number of pixels to zoom in')
    parser.add_argument('--tonumpy',action='store_true',help='Save image to a numpy file')
    parser.add_argument('--inflate',action='store_true',help='Inflates image to size zoom_pix x zoom_pix')
    args = parser.parse_args()

    generate_src_images(args.dataset_dir, args.num_observations, args.noise_std_low, args.noise_std_high, args.exclude_antenna_percentage, args.HA_point, args.HA_source, args.Dec_point, args.Dec_source, args.spectral_index_low, args.spectral_index_high, args.zoom_pix,args.tonumpy,args.inflate)

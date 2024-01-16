import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  
from nsfrb.simulating import compute_uvw, add_complex_gaussian_noise, simulate_far_field_rfi, rfi_source_position, simulate_near_field_rfi, get_core_coordinates, apply_phase_shift
from nsfrb.imaging import uniform_image
from nsfrb.config import NUM_CHANNELS, CH0, CH_WIDTH, AVERAGING_FACTOR, IMAGE_SIZE, c

def generate_rfi_images(dataset_dir, num_observations, ha_low, ha_high, dec_low, dec_high, rfi_prob, azimuth_low, azimuth_high, elevation_low, elevation_high, dist_low, dist_high, zoom_pix):
    """
    Generate images of RFI using near field or far field simulations and save the observations.
    It takes various parameters such as the dataset directory, the number of observations, 
    the bounds for the Hour Angle (HA) and Declination (Dec), the probability of adding RFI, 
    the bounds for the azimuth and elevation angles for near-field RFI, the bounds for the distance 
    for near-field RFI, and the zoom level for the generated images.

    Args:
        dataset_dir (str): The directory where the dataset will be saved.
        num_observations (int): The number of observations to generate.
        ha_low (float): The lower bound of the Hour Angle.
        ha_high (float): The upper bound of the Hour Angle.
        dec_low (float): The lower bound of the Declination.
        dec_high (float): The upper bound of the Declination.
        rfi_prob (float): The probability of adding RFI.
        azimuth_low (float): The lower bound of the azimuth angle for near-field RFI.
        azimuth_high (float): The upper bound of the azimuth angle for near-field RFI.
        elevation_low (float): The lower bound of the elevation angle for near-field RFI.
        elevation_high (float): The upper bound of the elevation angle for near-field RFI.
        dist_low (float): The lower bound of the distance for near-field RFI.
        dist_high (float): The upper bound of the distance for near-field RFI.
        zoom_pix (int): The number of pixels to zoom in on the generated images.

    Returns:
        None
    """
    dataset_dir = os.path.join(dataset_dir, 'rfi_examples')

    x_core, y_core, z_core = get_core_coordinates()
    pixel_resolution = (0.20 / np.max(np.sqrt(x_core**2 + y_core**2))) / 3

    for obs in tqdm(range(num_observations), desc="Progress"):  # Use tqdm for progress bar
        # Randomizing the Hour Angle and Declination
        HA = np.random.uniform(low=ha_low, high=ha_high)
        Dec = np.random.uniform(low=dec_low, high=dec_high)

        # Calculate u,v,w coordinates
        u_core, v_core, w_core = compute_uvw(x_core, y_core, z_core, HA, Dec)
        num_baselines = len(u_core)

        V = [np.zeros(num_baselines) + 1j*np.zeros(num_baselines) for _ in range(NUM_CHANNELS)]
        V_noisy = [add_complex_gaussian_noise(v, std_dev=0.1) for v in V]

        # RFI Types
        rfi_types = ["far", "near"]

        # Randomly select the RFI types to be added
        selected_rfi_types = np.random.choice(rfi_types, size=np.random.randint(1, 2), replace=False, p=[rfi_prob, 1-rfi_prob])
        selected_rfi_types = selected_rfi_types.tolist()

        for rfi_type in selected_rfi_types:
            if rfi_type in ["far"]:
                # Applying the RFI and random shifts
                rfi_visibilities = simulate_far_field_rfi(u_core, v_core, NUM_CHANNELS, num_baselines, pixel_resolution)
                V_noisy = [v + rfi_v for v, rfi_v in zip(V_noisy, rfi_visibilities)]
            else:
                azimuth = np.random.uniform(azimuth_low, azimuth_high)  # Specify azimuth in degrees
                elevation = np.random.uniform(elevation_low, elevation_high)  # Elevation
                dist = np.random.uniform(dist_low, dist_high)
                rfi_position = rfi_source_position(x_core, y_core, z_core, azimuth, elevation, distance=dist)

                # Simulate near-field RFI
                rfi_amplitude = 100
                for channel in range(NUM_CHANNELS):
                    frequency = (CH0 + CH_WIDTH * channel) * 1e6  # Calculate frequency for current channel
                    V_noisy[channel] = simulate_near_field_rfi(x_core, y_core, z_core, V_noisy[channel], rfi_position, rfi_amplitude, frequency)

        dirty_images = []
        for i in range(0, NUM_CHANNELS, AVERAGING_FACTOR):
            chunk_V = V_noisy[i:i+AVERAGING_FACTOR]
            avg_freq = CH0 + CH_WIDTH * i + AVERAGING_FACTOR/2 * CH_WIDTH

            wavelength = c / (avg_freq * 1e6)
            chunk_u_core = u_core / wavelength
            chunk_v_core = v_core / wavelength

            u_shift_rad, v_shift_rad = 0, 0
            # Apply the phase shift to chunk_V
            chunk_V_shifted = [apply_phase_shift(v, u_core, v_core, u_shift_rad, v_shift_rad) for v in chunk_V]

            dirty_img = uniform_image(chunk_V_shifted, chunk_u_core, chunk_v_core, IMAGE_SIZE)
            dirty_images.append(dirty_img)

        # Creating necessary directories
        observation_dir = os.path.join(dataset_dir, f'observation_{obs+1}/images')
        os.makedirs(observation_dir, exist_ok=True)

        # Metadata
        metadata = []

        # Saving the images and collecting metadata
        for i, dirty_img in enumerate(dirty_images):
            avg_freq = CH0 + CH_WIDTH * i * AVERAGING_FACTOR
            filename = f'subband_avg_{avg_freq:.2f}_MHz.png'
            filepath = os.path.join(observation_dir, filename)
            im_zoom = np.array(np.fliplr(np.abs(dirty_img.T)))
            im_zoom = im_zoom[(IMAGE_SIZE//2-zoom_pix):(IMAGE_SIZE//2+zoom_pix),(IMAGE_SIZE//2-zoom_pix):(IMAGE_SIZE//2+zoom_pix)]
            plt.imsave(filepath, im_zoom, cmap='gray')

        metadata.append({
            'selected_rfi_types': selected_rfi_types,
            'HA': HA,
            'Dec': Dec,
        })

        # Saving metadata to a CSV file
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(dataset_dir, f'observation_{obs+1}/metadata.csv'), index=False)

    print("All observations saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images of RFI observed with DSA-110 core antennas.')
    parser.add_argument('--dataset_dir', type=str, default='/Users/nikita/dsa110-nsfrb//simulations_and_classifications', help='Dataset directory')
    parser.add_argument('--num_observations', type=int, default=5, help='Number of generated observations')
    parser.add_argument('--ha_low', type=float, default=0, help='Lower bound for Hour Angle')
    parser.add_argument('--ha_high', type=float, default=5/180*np.pi, help='Upper bound for Hour Angle')
    parser.add_argument('--dec_low', type=float, default=-np.pi/2, help='Lower bound for Declination')
    parser.add_argument('--dec_high', type=float, default=np.pi/2, help='Upper bound for Declination')
    parser.add_argument('--rfi_prob', type=float, default=0.5, help='Probability p of selecting far field RFI type (near field will be 1-p))')
    parser.add_argument('--azimuth_low', type=float, default=0, help='Lower bound for azimuth')
    parser.add_argument('--azimuth_high', type=float, default=360, help='Upper bound for azimuth')
    parser.add_argument('--elevation_low', type=float, default=0, help='Lower bound for elevation')
    parser.add_argument('--elevation_high', type=float, default=10, help='Upper bound for elevation')
    parser.add_argument('--dist_low', type=float, default=1e3, help='Lower bound for distance (in meters)')
    parser.add_argument('--dist_high', type=float, default=1e7, help='Upper bound for distance (in meters)')
    parser.add_argument('--zoom_pix', type=int, default=25, help='Number of pixels for zooming')
    args = parser.parse_args()

    generate_rfi_images(args.dataset_dir, args.num_observations, args.ha_low, args.ha_high, args.dec_low, args.dec_high, args.rfi_prob, args.azimuth_low, args.azimuth_high, args.elevation_low, args.elevation_high, args.dist_low, args.dist_high, args.zoom_pix)


import numpy as np
from generate_source_images import generate_src_images
import os
from nsfrb.config import gridsize
"""
Generate PSF images for declinations spaced by the instantaneous FOV (3 degrees)
"""

#average FOV
FOV = gridsize*(36/3600) #~3 degrees
decs = np.arange(-90,90,FOV)

#output dir
dataset_dir = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-PSF/"
num_observations = 1
noise_std_low = noise_std_high = 0 #noiseless
exclude_antenna_percentage = (0,0) #ideally have all antennas
HA_low = HA_high = 0 #shouldn't vary with HA
spectral_index_low = spectral_index_high = 0
zoom_pix = gridsize*2
tonumpy = True

for dec in decs:
    Dec_low = Dec_high = dec
    generate_src_images(dataset_dir, num_observations, noise_std_low, noise_std_high, exclude_antenna_percentage, HA_low, HA_high, Dec_low, Dec_high, spectral_index_low, spectral_index_high, zoom_pix, tonumpy)
    #move to top level for ease of access
    outpath = os.path.join(dataset_dir, f'src_examples/observation_1/images/final_img_{HA_low:.2f}_hr_{dec:.2f}_deg.npy')
    print("mv " + outpath + " " + dataset_dir + "PSF_" + str(gridsize) + f"_{dec:.2f}_deg.npy")
    os.system("mv " + outpath + " " + dataset_dir + "PSF_" + str(gridsize) + f"_{dec:.2f}_deg.npy")

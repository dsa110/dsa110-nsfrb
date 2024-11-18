import argparse
import numpy as np
from simulations_and_classifications.generate_source_images import generate_src_images
import os
from nsfrb.config import gridsize,pixsize,IMAGE_SIZE
"""
Generate PSF images for declinations spaced by the instantaneous FOV (3 degrees)
"""


#simple wrapper function to make single PSF
def generate_PSF_images(dataset_dir,dec,zoom_pix,tonumpy,nsamps=1,dtype=np.float32,HA=0,injectnoise=0,noise_only=False,srcDECoffset=0,srcHAoffset=0):
    num_observations = 1
    noise_std_low = injectnoise#0 #noiseless
    noise_std_high = injectnoise
    exclude_antenna_percentage = (0,0) #ideally have all antennas
    HA_low = HA_high = HA #shouldn't vary with HA

    HA_point = HA
    HA_source= HA + srcHAoffset
    Dec_point = dec
    Dec_source = dec + srcDECoffset

    spectral_index_low = spectral_index_high = 0
    tonumpy = True
    print("generating PSF with ",nsamps,"samples")
    PSF = np.array(generate_src_images(dataset_dir, num_observations, noise_std_low, noise_std_high, exclude_antenna_percentage, HA_point, HA_source, Dec_point, Dec_source, spectral_index_low, spectral_index_high, zoom_pix, tonumpy,inflate=zoom_pix>IMAGE_SIZE//2,noise_only=noise_only,N_NOISE=nsamps),dtype=dtype)
    print("newshape:",PSF.shape)
    return PSF

#average FOV
FOV = gridsize*(36/3600) #~3 degrees
decs = np.arange(-90,90,FOV)

#output dir
num_observations = 1
noise_std_low = noise_std_high = 0 #noiseless
exclude_antenna_percentage = (0,0) #ideally have all antennas
HA_low = HA_high = 0 #shouldn't vary with HA
HAs = [0]
spectral_index_low = spectral_index_high = 0
tonumpy = True

def main(args):
    zoom_pix = args.gridsize//2
    print("mkdir " + args.dataset_dir + "gridsize_" + str(args.gridsize))
    os.system("mkdir " + args.dataset_dir + "gridsize_" + str(args.gridsize))

    for dec in decs:
        for HA in HAs:
            generate_PSF_images(args.dataset_dir,dec,zoom_pix,tonumpy,HA=HA)
            #Dec_low = Dec_high = dec
            #generate_src_images(args.dataset_dir, num_observations, noise_std_low, noise_std_high, exclude_antenna_percentage, HA_low, HA_high, Dec_low, Dec_high, spectral_index_low, spectral_index_high, zoom_pix, tonumpy)
            #move to top level for ease of access
            outpath = os.path.join(args.dataset_dir, f'src_examples/observation_1/images/final_img_{HA:.2f}_hr_{dec:.2f}_deg.npy')
            print("mv " + outpath + " " + args.dataset_dir + "gridsize_" + str(args.gridsize) + "/PSF_" + str(args.gridsize) + f"_{dec:.2f}_deg.npy")
            os.system("mv " + outpath + " " + args.dataset_dir + "gridsize_" + str(args.gridsize) + "/PSF_" + str(args.gridsize) + f"_{dec:.2f}_deg.npy")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate PSFs at all declinations with DSA-110 core antennas.')
    parser.add_argument('--dataset_dir', type=str, default="/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-PSF/", help='Dataset directory')
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, default=300',default=300)
    args = parser.parse_args()

    main(args)

import glob
import argparse
import numpy as np
from simulations_and_classifications.generate_source_images import generate_src_images
import os
from nsfrb.outputlogging import printlog
from nsfrb.config import gridsize,pixsize,IMAGE_SIZE,psf_dir,processfile,nsamps,flagged_antennas,bmin,robust
"""
Generate PSF images for declinations spaced by the instantaneous FOV (3 degrees)
"""


#simple wrapper function to make single PSF
def generate_PSF_images(dataset_dir,dec,zoom_pix,tonumpy,nsamps=1,dtype=np.float32,HA=0,injectnoise=0,noise_only=False,srcDECoffset=0,srcHAoffset=0,flagged_antennas=flagged_antennas,bmin=bmin,robust=robust,xidxs=None,yidxs=None):
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
    PSF = np.array(generate_src_images(dataset_dir, num_observations, noise_std_low, noise_std_high, exclude_antenna_percentage, HA_point, HA_source, Dec_point, Dec_source, spectral_index_low, spectral_index_high, zoom_pix, tonumpy,inflate=zoom_pix>IMAGE_SIZE//2,noise_only=noise_only,N_NOISE=nsamps,flagged_antennas=flagged_antennas,bmin=bmin,robust=robust,xidxs=xidxs,yidxs=yidxs),dtype=dtype)
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


def make_PSF_dict(PSF_dict=dict()):
    """
    This function returns a dictionary of available PSF files that have already been generated
    """
    psfnames = glob.glob(psf_dir+"gridsize_*")
    for psfname in psfnames:
        gsize = int(psfname[psfname.index("gridsize_")+9:])
        PSF_dict[gsize] = dict()
        PSF_decs = []
        decnames = glob.glob(psfname+"/*npy")
        for decname in decnames:
            dec = float(decname[decname.index("PSF_")+ 5 +len(str(gsize)):decname.index("_deg")])
            PSF_decs.append(float(decname[decname.index("PSF_") + 5 + len(str(gsize)):decname.index("_deg")]))

        PSF_dict[gsize]['declabels'] = np.array(np.sort(PSF_decs))
    """
    if gridsize in PSF_dict.keys():
        printlog("loading PSF for gridsize " + str(gridsize) +", declination " + str(PSF_dict[gridsize]['declabels'][np.argmin(np.abs(PSF_dict[gridsize]['declabels']-np.nanmean(DEC_axis)))]),output_file=processfile)
        default_PSF = np.array(np.load(psf_dir + "gridsize_" + str(gridsize) + "/PSF_" + str(gridsize) + "_{d:.2f}".format(d=PSF_dict[gridsize]['declabels'][np.argmin(np.abs(PSF_dict[gridsize]['declabels']-np.nanmean(DEC_axis)))]) + "_deg.npy"),dtype=np.float32)[:,:,np.newaxis,:].repeat(nsamps,axis=2)
        default_PSF_params = (gridsize,PSF_dict[gridsize]['declabels'][np.argmin(np.abs(PSF_dict[gridsize]['declabels']-np.nanmean(DEC_axis)))])
    else:
        default_PSF = scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),gridsize//2,True,nsamps)#sim.make_PSF_cube()
        default_PSF_params = (gridsize,np.nanmean(DEC_axis))
    """
    return PSF_dict

def save_PSF(PSF,kernel_size,dec):
    os.system("mkdir " + psf_dir + "gridsize_" + str(kernel_size) )
    np.save(psf_dir + "gridsize_" + str(kernel_size)+"/PSF_" + str(kernel_size) + "_{d:.2f}".format(d=dec) + "_deg.npy",PSF[:,:,0,:].astype(np.float32))
    return psf_dir + "gridsize_" + str(kernel_size)+"/PSF_" + str(kernel_size) + "_{d:.2f}".format(d=dec) + "_deg.npy"


def manage_PSF(PSF_dict,kernel_size,dec,default_PSF_params=(-1,180),default_PSF=None,nsamps=nsamps):
    """
    This function handles searching for available PSFs on disk and generating new PSFs if none 
    are available to match the desired gridsize and declination
    """

    if default_PSF_params[0] == kernel_size: #CASE1: default has right kernel size
        if np.abs(default_PSF_params[1] - dec)<1.5:#Case1A: default has right dec
            printlog("PSF is valid:" + str(default_PSF_params),output_file=processfile)# don't do anything
        else: #Case1B: default has wrong dec
            if kernel_size in PSF_dict.keys() and np.abs(PSF_dict[kernel_size]['declabels'][np.argmin(np.abs(PSF_dict[kernel_size]['declabels']-dec))]-dec)<1.5: #Case1Bx: psf_dict has right kernel size and dec
                best_dec = PSF_dict[kernel_size]['declabels'][np.argmin(np.abs(PSF_dict[kernel_size]['declabels']-dec))] #pull PSF from file
                printlog("loading PSF for kernelsize " + str(kernel_size) +", declination " + str(best_dec),output_file=processfile)
                default_PSF = np.array(np.load(psf_dir + "gridsize_" + str(kernel_size) + "/PSF_" + str(kernel_size) + "_{d:.2f}".format(d=best_dec) + "_deg.npy"),dtype=np.float32)[:,:,np.newaxis,:].repeat(nsamps,axis=2)
                default_PSF_params = (kernel_size,best_dec)
            else: #Case1By: psf_dict does not have right kerenel size and dec
                printlog("making PSF for kernelsize " + str(kernel_size) + ", declination " + "{d:.2f}".format(d=dec),output_file=processfile) #generate new PSF
                default_PSF = generate_PSF_images(psf_dir,dec,kernel_size,True,nsamps)
                default_PSF_params =  (kernel_size,float("{d:.2f}".format(d=dec)))
                printlog("Saved PSF to " + str(save_PSF(default_PSF,kernel_size,dec)),output_file=processfile)
                printlog("updating PSF dict...",output_file=processfile)
                make_PSF_dict(PSF_dict)
    else: #CASE2: default has wrong kernel size
        if kernel_size in PSF_dict.keys() and np.abs(PSF_dict[kernel_size]['declabels'][np.argmin(np.abs(PSF_dict[kernel_size]['declabels']-dec))]-dec)<1.5: #Case1Bx: psf_dict has right kernel size and dec
            best_dec = PSF_dict[kernel_size]['declabels'][np.argmin(np.abs(PSF_dict[kernel_size]['declabels']-dec))] #pull PSF from file
            printlog("loading PSF for kernelsize " + str(kernel_size) +", declination " + str(best_dec),output_file=processfile)
            default_PSF = np.array(np.load(psf_dir + "gridsize_" + str(kernel_size) + "/PSF_" + str(kernel_size) + "_{d:.2f}".format(d=best_dec) + "_deg.npy"),dtype=np.float32)[:,:,np.newaxis,:].repeat(nsamps,axis=2)
            default_PSF_params = (kernel_size,best_dec)
        else: #Case1By: psf_dict does not have right kerenel size and dec
            printlog("making PSF for kernelsize " + str(kernel_size) + ", declination " + "{d:.2f}".format(d=dec),output_file=processfile) #generate new PSF
            default_PSF = generate_PSF_images(psf_dir,dec,kernel_size,True,nsamps)
            default_PSF_params =  (kernel_size,float("{d:.2f}".format(d=dec)))
            printlog("Saved PSF to " + str(save_PSF(default_PSF,kernel_size,dec)),output_file=processfile)
            printlog("updating PSF dict...",output_file=processfile)
            make_PSF_dict(PSF_dict)
    return default_PSF,default_PSF_params



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate PSFs at all declinations with DSA-110 core antennas.')
    parser.add_argument('--dataset_dir', type=str, default="/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-PSF/", help='Dataset directory')
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, default=300',default=300)
    args = parser.parse_args()

    main(args)

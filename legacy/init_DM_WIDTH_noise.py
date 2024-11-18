from matplotlib import pyplot as plt
import glob
import jax
import jax.numpy as jnp
import numpy as np
from nsfrb.config import *
from nsfrb.searching import DM_trials,widthtrials,get_RA_cutoff,corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append,maxshift,full_boxcar_filter
from nsfrb import noise
from nsfrb import simulating
from inject import injecting
import argparse
from simulations_and_classifications import generate_PSF_images as scPSF
import os
import sys
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")

#output_dir = cwd + "/tmpoutput/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/"
coordfile = cwd + "/DSA110_Station_Coordinates.csv" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/DSA110_Station_Coordinates.csv"
output_file = cwd + "-logfiles/search_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt"
cand_dir = cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
processfile = cwd + "-logfiles/process_log.txt"
frame_dir = cwd + "-frames/"
psf_dir = cwd + "-PSF/"
f=open(output_file,"w")
f.close()


flagged_antennas = [21, 22, 23, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 117]
"""
This script initializes the noise dict, whic varies with DM and width
"""


#jax noise estimation per DM and boxcar trial
@jax.jit
def noise_estimate(image_tesseract,PSFimg,corr_shifts_all,tdelays_frac,boxcar,noise,past_noise_N,noiseth):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """
    #matched filter
    truensamps = boxcar.shape[3]
    gridsize_DEC,gridsize_RA = image_tesseract.shape[:2]
    padby_DEC = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA = (gridsize_RA - PSFimg.shape[1])//2
    """
    gridsize = image_tesseract.shape[0]
    padby = (gridsize - PSFimg.shape[0])//2
    """
    #image_tesseract_point = jnp.concatenate([image_tesseract[:,:,:-truensamps,:],jnp.pad(jnp.real(jnp.fft.ifft2(jnp.fft.fft2(image_tesseract[:,:,-truensamps:,:],axes=(0,1),s=image_tesseract.shape[:2])*jnp.fft.fft2(PSFimg,axes=(0,1),s=image_tesseract.shape[:2]),axes=(0,1),s=image_tesseract.shape[:2])),((0,padby),(0,padby),(0,0),(0,0)))[-gridsize:,-gridsize:,:,:]],axis=2)
    #image_tesseract_point = jnp.concatenate([image_tesseract[:,:,:-truensamps,:],jnp.real(jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.fft2(image_tesseract[:,:,-truensamps:,:],axes=(0,1),s=image_tesseract.shape[:2])*jnp.fft.fft2(jnp.pad(PSFimg,((padby,padby),(padby,padby),(0,0),(0,0))),axes=(0,1),s=image_tesseract.shape[:2]),axes=(0,1),s=image_tesseract.shape[:2]),axes=(0,1)))],axis=2)


    #since we combined matched filtering, etc, we need to filter the full combined image
    """
    image_tesseract_point = jnp.real(jnp.fft.fftshift(
                                                jnp.fft.ifft2(
                                                    jnp.fft.fft2(jnp.fft.ifftshift(image_tesseract,axes=(0,1)),
                                                        axes=(0,1),s=(gridsize,gridsize))*jnp.fft.fft2(jnp.pad(
                                                            (jnp.fft.ifftshift(PSFimg.repeat(image_tesseract.shape[2],axis=2),axes=(0,1))),
                                                            ((padby,padby),(padby,padby),(0,0),(0,0))),
                                                            axes=(0,1),s=(gridsize,gridsize))
                                                    ,axes=(0,1),s=(gridsize,gridsize))
                                                ,axes=(0,1)))
    """
    image_tesseract_point = jnp.real(jnp.fft.fftshift(
                                                jnp.fft.ifft2(
                                                    jnp.fft.fft2(jnp.fft.ifftshift(image_tesseract,axes=(0,1)),
                                                        axes=(0,1),s=(gridsize_DEC,gridsize_RA))*jnp.fft.fft2(jnp.pad(
                                                            (jnp.fft.ifftshift(PSFimg.repeat(image_tesseract.shape[2],axis=2),axes=(0,1))),
                                                            ((padby_DEC,padby_DEC),(padby_RA,padby_RA),(0,0),(0,0))),
                                                            axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                                    ,axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                                ,axes=(0,1)))

    del PSFimg

    #dedispersion
    nsamps = image_tesseract_point.shape[-2]
    nDM = tdelays_frac.shape[3]

    image_tesseract_filtered_dm = (((jnp.take_along_axis(image_tesseract_point[:,:,:,jnp.newaxis,:].repeat(nDM,axis=3).repeat(2,axis=4),indices=corr_shifts_all,axis=2))*tdelays_frac).sum(4))[:,:,-truensamps:,:]

    del tdelays_frac
    del corr_shifts_all

    #boxcar filter
    image_tesseract_binned = jnp.nan_to_num(jnp.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(image_tesseract_filtered_dm,
                                                            n=image_tesseract_filtered_dm.shape[2],
                                                            axis=2,norm='backward')*jnp.fft.fft(boxcar,
                                                            n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),
                                                        n=image_tesseract_filtered_dm.shape[2],
                                                        axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)),
                                            nan=0,posinf=0,neginf=0)##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps

    del image_tesseract_filtered_dm
    #del boxcar

    #create masks
    mask = jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf

    #compute noise and update
    noise = noise.at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned,axis=4,where=mask
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))


    del mask
    del boxcar
    
    del image_tesseract_binned
    return jax.device_put(noise,jax.devices("cpu")[0])

def getnoise(DEC):
    

    #create injection with snr = 0, dm = 0, width = 0, HA =0 , DEC =0 
    HA = 0
    SNR = 0
    num_chans = int(NUM_CHANNELS//AVERAGING_FACTOR)
    noise_img=injecting.generate_inject_image(HA=HA,DEC=DEC,offsetRA=0,offsetDEC=0,snr=SNR,loc=0.5,gridsize=IMAGE_SIZE,nchans=num_chans,nsamps=nsamps+maxshift,DM=0,offline=False,noiseless=False) 

    """
    plt.figure()
    plt.imshow(noise_img.mean((0,1)).transpose(),aspect='auto')
    plt.savefig('testfig.png')
    plt.close()
    """

    #make a psf
    kernel_size = IMAGE_SIZE
    PSF_dict = dict()
    PSF_decs = []
    psfnames = glob.glob(psf_dir+"gridsize_*")
    for psfname in psfnames:
        gsize = int(psfname[psfname.index("gridsize_")+9:])
        PSF_dict[gsize] = dict()
        PSF_decs = []
        decnames = glob.glob(psfname+"/*npy")
        for decname in decnames:
            dec = float(decname[decname.index("PSF_"+str(gsize))+8:decname.index("_deg")])
            PSF_decs.append(float(decname[decname.index("PSF_"+str(gsize))+8:decname.index("_deg")]))

        PSF_dict[gsize]['declabels'] = np.array(np.sort(PSF_decs))
    print(PSF_dict[kernel_size].keys())
    if kernel_size in PSF_dict.keys():
        best_dec = PSF_dict[kernel_size]['declabels'][np.argmin(np.abs(PSF_dict[kernel_size]['declabels']-DEC))]
        print("loading PSF for kernelsize " + str(kernel_size) +", declination " + str(best_dec))
        default_PSF = np.array(np.load(psf_dir + "gridsize_" + str(kernel_size) + "/PSF_" + str(kernel_size) + "_{d:.2f}".format(d=best_dec) + "_deg.npy"),dtype=np.float32)[:,:,np.newaxis,:].repeat(nsamps,axis=2)
    elif kernel_size not in PSF_dict.keys() and (default_PSF_params[0] != kernel_size or np.abs(default_PSF_params[1] - float("{d:.2f}".format(d=DEC)))>1.5):
        print("making PSF for kernelsize " + str(kernel_size) + ", declination " + "{d:.2f}".format(d=DEC))
        default_PSF = scPSF.generate_PSF_images(psf_dir,DEC,kernel_size//2,True,nsamps)

    #cutoff
    #create offset grid using pixel size and max UV diagonal distance
    x_m,y_m,z_m = simulating.get_core_coordinates(flagged_antennas) #meters
    U,V,W = simulating.compute_uvw(x_m,y_m,z_m,0,DEC*np.pi/180) #meters
    uv_diag = np.max(np.sqrt(U**2 + V**2)) #meters
    pixel_resolution = (0.20 / uv_diag) / 3 #radians
    offset_grid = np.arange(-IMAGE_SIZE//2,IMAGE_SIZE//2)*pixel_resolution*180/np.pi #degrees

    print(offset_grid)
    assert(len(offset_grid) == IMAGE_SIZE)

    #add offset from image center
    #RA_axis = RA + offset_grid
    DEC_axis = DEC + offset_grid
    RA_cutoff=get_RA_cutoff(DEC_axis[len(DEC_axis)//2],pixsize=DEC_axis[1]-DEC_axis[0])
    noise_img = noise_img[:,:-RA_cutoff,:,:]
    default_PSF = default_PSF[:,:-RA_cutoff,:,:]

    #get noise estimate
    jaxdev = 0
    prev_noise,prev_noise_N = np.zeros((len(widthtrials),len(DM_trials))),0
    noiseth = 0
    output_noise = np.array(noise_estimate(jax.device_put(np.array(noise_img,dtype=np.float32),jax.devices()[jaxdev]),
                                                                 jax.device_put(np.array(default_PSF[:,:,0:1,:],dtype=np.float32),jax.devices()[jaxdev]),
                                                                 jax.device_put(corr_shifts_all_append,jax.devices()[jaxdev]),
                                                                 jax.device_put(tdelays_frac_append,jax.devices()[jaxdev]),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[jaxdev]),
                                                                 jax.device_put(np.array(prev_noise,dtype=np.float16),jax.devices()[jaxdev]),
                                                                 prev_noise_N,noiseth))
    print("Output noise:",output_noise,output_noise.shape)


    return output_noise


def main(args):

    DECS = np.arange(0,180,1.6)
    allnoise = np.zeros((5,16))
    for DEC in DECS:
        allnoise += getnoise(DEC)
    allnoise = allnoise/len(DECS)

    noise.noise_update_all(allnoise,IMAGE_SIZE,IMAGE_SIZE,DM_trials,widthtrials,writeonly=True,suff="_sim")
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Form and send dirty images extracting visibilities from raw data (.out) file.')
    """
    parser.add_argument('filelabel')           # positional argument
    parser.add_argument('timestamp',help='Timestamp in ISOT format (e.g. 2024-06-12T21:35:49)')
    parser.add_argument('--num_gulps', type=int, help='Number of gulps, default -1 for all ',default=-1)
    parser.add_argument('--num_time_samples', type=int, default=25, help='Number of time samples to extract from the .out file.')
    #parser.add_argument('--fringestop', action='store_true', default=False, help='Fringe stop manually')
    #parser.add_argument('--fringetable',type=str,help='Fringe stop manually with specified table in the dsa110-nsfrb-fast-visibilities dir',default='')
    parser.add_argument('--datasize',type=int,help='Data size in bytes, default=4',default=4)
    parser.add_argument('--path',type=str,help='Path to raw data files',default=vispath)
    parser.add_argument('--outpath',type=str,help='Output path for images',default=imgpath)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--search', action='store_true', default=False, help='Send resulting image to process server')
    parser.add_argument('--save',action='store_true',default=False,help='Save image as a numpy and fits file')
    parser.add_argument('--inject',action='store_true',default=False,help='Inject a burst into the gridded visibilities. Unless the --solo_inject flag is set, a noiseless injection will be integrated into the data.')
    parser.add_argument('--solo_inject',action='store_true',default=False,help='If set, visibility data will be zeroed and an injection with simulated noise will overwrite the data')
    parser.add_argument('--snr_inject',type=float,help='SNR of injection; default 0 which chooses a random SNR',default=0)
    parser.add_argument('--dm_inject',type=float,help='DM of injection; default -1 which chooses a random DM',default=-1)
    parser.add_argument('--width_inject',type=int,help='Width of injection in samples; default 0 which chooses a random width',default=0)
    parser.add_argument('--offline',action='store_true',default=False,help='Initializes previous frame with noise')
    """
    args = parser.parse_args()
    main(args)

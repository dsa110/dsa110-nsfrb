import numpy as np
from nsfrb import config
from astropy.time import Time
from nsfrb.searching import maxshift
import copy
from nsfrb import jax_funcs
from nsfrb.planning import uv_to_pix
from inject import injecting
from nsfrb import jax_funcs
import jax

from nsfrb.searching import corr_shifts_all_append, tdelays_frac_append,full_boxcar_filter,DM_trials
import jax.numpy as jnp
#get all fast vis dates
import numpy as np
from nsfrb import pipeline,config,planning
import glob
from matplotlib import pyplot as plt
from astropy.time import Time
import os

usedev=0
gpdir = "/dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/sensitivitydata/"
allfs = np.sort(glob.glob(gpdir+"/*sb01*"))
allinfo = []
allisots = []
alldecs = []
allkeeps = []
for i in range(len(allfs)):
    #check that all files are there
    if len(glob.glob(gpdir + os.path.basename(allfs[i])[:-9] + "*"))==16:
        allkeeps.append(i)
        info=pipeline.read_raw_vis(allfs[i],get_header=True)
        #if np.abs(info[1]-61092.86449934028)<(2*60/86400):
        print("good--",info)
        allinfo.append(tuple(list(info)+[allfs[i]]))
        allisots.append(os.path.basename(allinfo[-1][-1])[:-9])
    else:
        print("bad--",allfs[i])
alltimes = Time([allinfo[i][1] for i in range(len(allinfo))],format='mjd')
allfs = allfs[np.array(allkeeps)]
print("final list: "+str(len(allfs)))


HA = 0
Dec = 16.27
offsetRA = offsetDEC = 0
SNR = 6
width=1
DM=0
_offset=i=20
gridsize=175
noise = np.load(config.img_dir + "senstest_noise.npy")
t_now = Time(allisots[_offset],format='isot')
mjd = t_now.mjd
time_start_isot = t_now.isot
RA_axis,Dec_axis,elev = uv_to_pix(mjd,gridsize,two_dim=False,manual=False,DEC=Dec)
HA_axis = RA_axis - RA_axis[int(gridsize//2)]
injloc=0.5

noise_inj = np.load(config.img_dir + "senstest_noise.npy") #np.zeros(5)
past_noise_N=0
noiseth=3
RA_cutoff = planning.get_RA_cutoff(Dec,T=(config.tsamp)*config.nsamps)

dirty_img = np.load(config.img_dir +"_senstestimage.npy",mmap_mode='r')

nsamps=25
normax=np.zeros((5,16))
for wi in range(5):
    injloc = 0.5 - ((2**wi)/nsamps/2)
    for di in range(len(DM_trials)):
        inject_img = injecting.generate_inject_image(Time.now().isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,
                                             snr=SNR,width=int(2**wi),loc=injloc,gridsize=gridsize,nchans=16,
                                             nsamps=25,DM=DM_trials[di],maxshift=maxshift,offline=False,
                                             noiseless=True,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=False,
                                             bmin=config.bmin,robust= -2)

        tmpimg = np.nanmean(dirty_img[:,:,25*i:25*(i+1),:].reshape((gridsize,gridsize,25,16,8)),-1) + inject_img#*(SNR*noise[0]/np.nanmax(inject_img)))
        image_tesseract_filtered_cut = np.concatenate([np.nanmean(dirty_img[:,:-RA_cutoff,maxshift*(i-1):maxshift*i,:].reshape((gridsize,gridsize-RA_cutoff,maxshift,16,8)),-1),tmpimg[:,RA_cutoff:,:,:]],axis=2)
        image_tesseract_filtered_cut[np.isnan(image_tesseract_filtered_cut)] = 0
        """
        inject_img = injecting.generate_inject_image(Time.now().isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,
                                             snr=0.3*SNR*1e-9/4,width=int(2**wi),loc=injloc,gridsize=gridsize,nchans=16,
                                             nsamps=25,DM=DM_trials[di],maxshift=maxshift,offline=False,
                                             noiseless=False,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=False,
                                             bmin=config.bmin,robust= -2)
        inject_img_noise = injecting.generate_inject_image(Time.now().isot,HA=HA,DEC=Dec,offsetRA=offsetRA,offsetDEC=offsetDEC,
                                             snr=0.3*SNR*1e-9/4,width=int(2**wi),loc=injloc,gridsize=gridsize,nchans=16,
                                             nsamps=25,DM=DM_trials[di],maxshift=maxshift,offline=False,
                                             noiseless=False,HA_axis=HA_axis,DEC_axis=Dec_axis,noiseonly=True,
                                             bmin=config.bmin,robust= -2)

        image_tesseract_filtered_cut = np.concatenate([inject_img_noise[:,:-RA_cutoff,-maxshift:,:].reshape((gridsize,gridsize-RA_cutoff,maxshift,16)),inject_img[:,RA_cutoff:,:,:]],axis=2)
        """
        image_tesseract_filtered_cut[np.isnan(image_tesseract_filtered_cut)] = 0


        image_tesseract_binned_new,noise_inj,tmp = jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                      jax.device_put(np.array(dirty_img[:,:,0:1,0],dtype=bool),jax.devices()[usedev]),
                                      jax.device_put(corr_shifts_all_append,jax.devices()[usedev]),
                                      jax.device_put(tdelays_frac_append,jax.devices()[usedev]),
                                      jax.device_put(full_boxcar_filter,jax.devices()[usedev]),
                                      jax.device_put(np.zeros_like(noise),jax.devices()[usedev]),past_noise_N,noiseth)
        print(image_tesseract_binned_new[90,87,wi,di])
        normax[wi,di] = (image_tesseract_binned_new[90,87,wi,di])
        print(noise)
        print(noise_inj)
np.save("normax6.npy",normax)

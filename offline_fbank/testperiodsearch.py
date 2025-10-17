from nsfrb import jax_funcs
import numpy as np
import jax
from nsfrb import periodicity
from nsfrb.searching import gen_dm_shifts,gen_dm,gen_boxcar_filter
from nsfrb.config import minDM,maxDM,DM_tol,fc,nchans,tsamp,chanbw,freq_axis,noise_dir



trial_p_samp = np.array([19,50,2250],dtype=int)
nsamps = 2250#2250
gridsize = 10#175
batches = 1#35
gridsize_batch = int(gridsize//batches)
image_tesseract_input = np.zeros((gridsize,gridsize,nsamps,16))
image_tesseract_input[gridsize//2,gridsize//2,np.arange(nsamps//trial_p_samp[0],dtype=int)*trial_p_samp[0],:] = 100
from scipy.stats import norm
image_tesseract_input += norm.rvs(loc=0,scale=1,size=image_tesseract_input.shape)



DM_trials = np.array(gen_dm(minDM,maxDM,DM_tol,fc*1e-3,nchans,tsamp,chanbw,nsamps))#[0:1]
nDMtrials = len(DM_trials)
corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append = gen_dm_shifts(DM_trials,freq_axis,tsamp,nsamps)

#make boxcar filters in advance
widthtrials = np.array(2**np.arange(5),dtype=int)
nwidths = len(widthtrials)

full_boxcar_filter = gen_boxcar_filter(widthtrials,nsamps)


trial_p_samp_idxs,trial_p_folds_factor = periodicity.gen_p_trials(nsamps,trial_p_samp)
gidxwidth,gidxdec,gidxra,gidxdm,gidxsamp = np.meshgrid(np.arange(nwidths,dtype=int),
                                                            np.arange(gridsize_batch,dtype=int),
                                                            np.arange(gridsize_batch,dtype=int),
                                                            np.arange(nDMtrials,dtype=int),
                                                            trial_p_samp_idxs.flatten())
                                                            #np.arange(len(trial_p_samp)))
tmp,tmp,tmp,tmp,tmp,gidxP = np.meshgrid(np.arange(nwidths,dtype=int),
                                                            np.arange(gridsize_batch,dtype=int),
                                                            np.arange(gridsize_batch,dtype=int),
                                                            np.arange(nDMtrials,dtype=int),
                                                            np.arange(nsamps,dtype=int),
                                                            np.arange(len(trial_p_samp)))
gidxwidth = gidxwidth.transpose((1,0,2,3,4)).reshape((nwidths,gridsize_batch,gridsize_batch,nDMtrials,nsamps,len(trial_p_samp)))
gidxdec = gidxdec.transpose((1,0,2,3,4)).reshape((nwidths,gridsize_batch,gridsize_batch,nDMtrials,nsamps,len(trial_p_samp)))
gidxra = gidxra.transpose((1,0,2,3,4)).reshape((nwidths,gridsize_batch,gridsize_batch,nDMtrials,nsamps,len(trial_p_samp)))
gidxdm = gidxdm.transpose((1,0,2,3,4)).reshape((nwidths,gridsize_batch,gridsize_batch,nDMtrials,nsamps,len(trial_p_samp)))
gidxsamp = gidxsamp.transpose((1,0,2,3,4)).reshape((nwidths,gridsize_batch,gridsize_batch,nDMtrials,nsamps,len(trial_p_samp)))
gidxP = gidxP.transpose((1,0,2,3,4,5))#.reshape((nwidths,gridsize_batch,gridsize_batch,nDMtrials,nsamps,len(trial_p_samp)))
#print(gidxwidth.shape,gidxwidth.flatten().shape)
#print(gidxP.shape,gidxP.flatten().shape)

import pickle as pkl
f = open(noise_dir+"/noise_175x175.pkl","rb")
p=pkl.load(f)
f.close()
pn = []
#from nsfrb.searching import DM_trials,widthtrials


#for k in p.keys():
for ki in p[0].keys():
    print(ki,p[0][ki][1])
    pn.append(p[0][ki][1])
pn = np.array(pn)


#nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps
import time
t1 = time.time()
imgout_full = np.zeros((gridsize,gridsize,nwidths,len(DM_trials),len(trial_p_samp)))
for i in range(batches):
    for j in range(batches):
        snrs = np.zeros((nwidths,gridsize_batch,gridsize_batch,len(DM_trials),nsamps,len(trial_p_samp)))
        inputnoise = pn #np.zeros((len(widthtrials),len(DM_trials)))
        imgout,noiseout,imgoutb,imgsnrs = jax_funcs.ffa_slow_jit(jax.device_put(image_tesseract_input[i*gridsize_batch:(i+1)*gridsize_batch,j*gridsize_batch:(j+1)*gridsize_batch,:,:],jax.devices()[0]),
                                         jax.device_put(snrs,jax.devices()[0]),
                                         jax.device_put(corr_shifts_all_no_append,jax.devices()[0]),
                                         jax.device_put(tdelays_frac_no_append,jax.devices()[0]),
                                         jax.device_put(full_boxcar_filter,jax.devices()[0]),
                                         jax.device_put(inputnoise,jax.devices()[0]),
                                         0,3,
                                         jax.device_put(gidxwidth.flatten(),jax.devices()[0]),
                                         jax.device_put(gidxdec.flatten(),jax.devices()[0]),
                                         jax.device_put(gidxra.flatten(),jax.devices()[0]),
                                         jax.device_put(gidxdm.flatten(),jax.devices()[0]),
                                         jax.device_put(gidxsamp.flatten(),jax.devices()[0]),
                                         jax.device_put(gidxP.flatten(),jax.devices()[0]),
                                         #jax.device_put(trial_p_samp_idxs[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:].repeat(nwidths,0).repeat(gridsize,1).repeat(gridsize,2).repeat(nDMtrials,3),jax.devices()[0]),
                                         jax.device_put(trial_p_folds_factor[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:].repeat(nwidths,0).repeat(gridsize_batch,1).repeat(gridsize_batch,2).repeat(nDMtrials,3),jax.devices()[0]))
        print(imgout.shape)
        imgout_full[i*gridsize_batch:(i+1)*gridsize_batch,j*gridsize_batch:(j+1)*gridsize_batch,:,:,:] =  imgout

print(time.time()-t1,'seconds')

np.save("testperiodimg.npy",imgout_full)
np.save("testinputimg.npy",image_tesseract_input)
np.save("testbinnedimg.npy",imgoutb)
np.save("testsnrs.npy",imgsnrs)
np.save("testnoise.npy",noiseout)

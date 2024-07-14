import jax
import jax.numpy as jnp
import numpy as np

"""
This file defines jit compiled functions to accelerate GPU compilation and computation.
"""

@jax.jit
def inner_snr_fft_jit(image_tesseract_filtered_dm,boxcar):
    """
    This function replaces pyptorch with JAX so that JIT computation can be invoked
    """
   


    #take fourier transform of boxcar and image
    image_tesseract_binned = jax.device_put(np.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(jnp.array(image_tesseract_filtered_dm),n=image_tesseract_filtered_dm.shape[2],axis=2,norm='backward')*jnp.fft.fft(jnp.array(boxcar),n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),
                                                                                                                                            n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)),jax.devices("cpu")[0]) ##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps
    del image_tesseract_filtered_dm
    del boxcar

    return image_tesseract_binned 
    
@jax.jit
def inner_snr_conv_jit(image_tesseract_filtered_dm,boxcar):
    """
    This function replaces pyptorch with JAX so that JIT computation can be invoked
    """



    #take fourier transform of boxcar and image
    
    
    image_tesseract_binned = jax.device_put(jnp.apply_along_axis(func1d=lambda x : jnp.convolve(x[:image_tesseract_filtered_dm.shape[2]],x[image_tesseract_filtered_dm.shape[2]:],mode='same'),axis=3,arr=jnp.concatenate([jnp.array(image_tesseract_filtered_dm[jnp.newaxis,:,:,:,:]).repeat(boxcar.shape[0],axis=0),jnp.array(boxcar)],axis=3)),jax.devices("cpu")[0])
   
    del image_tesseract_filtered_dm
    del boxcar

    return image_tesseract_binned

@jax.jit
def inner_dedisperse_jit(image_tesseract_point,DM_trials_in,tsamp,freq_axis_in):#,fout):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #make JAX arrays 
    #print(image_tesseract_point.shape) 
    #image_tesseract_point = image_tesseract_point#.numpy()
    DM_trials = jnp.array(DM_trials_in)
    freq_axis = jnp.array(freq_axis_in)

    #Delays
    gridsize = image_tesseract_point.shape[0]
    #print("SIZES:" + str(gridsize),file=fout)
    nchans = len(freq_axis)
    nsamps = image_tesseract_point.shape[-2]
    


    tdelays = -(((DM_trials[:,jnp.newaxis].repeat(nchans,axis=1))*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))).transpose())
    tdelays_idx_hi = jnp.array(jnp.ceil(tdelays/tsamp),dtype=jnp.int16)
    tdelays_idx_low = jnp.array(jnp.floor(tdelays/tsamp),dtype=jnp.int16)
    tdelays_frac = tdelays/tsamp - tdelays_idx_low

    #print("TDELHI_FREQ_DM:" + str(tdelays_idx_hi),file=fout)
    #print("TDELLOW_FREQ_DM:" + str(tdelays_idx_low),file=fout)
    #print("Trial DM: " + str(DM_trials.shape) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout)
   


    #rearrange shift idxs
    idxs_all = (np.arange(nsamps)[:,jnp.newaxis,jnp.newaxis]).repeat(nchans,axis=1).repeat(len(DM_trials),axis=2)
    corr_shifts_all_hi = np.clip(((-tdelays_idx_hi[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)#(idxs_all - shifts_all_hi)%nsamps
    corr_shifts_all_low = np.clip(((-tdelays_idx_low[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)#(idxs_all - shifts_all_low)%nsamps
    #print("TDEL_HI:" + str(corr_shifts_all_hi),file=fout)
    #print("TDEL_LOW:" + str(corr_shifts_all_low),file=fout)

    #apply delays
    tdelays_frac = tdelays_frac[jnp.newaxis,jnp.newaxis,jnp.newaxis,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1).repeat(nsamps,axis=2)
    corr_shifts_all_hi = corr_shifts_all_hi[jnp.newaxis,jnp.newaxis,:,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    corr_shifts_all_low = corr_shifts_all_low[jnp.newaxis,jnp.newaxis,:,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    image_tesseract_point_DM = image_tesseract_point[:,:,:,:,jnp.newaxis].repeat(len(DM_trials),axis=4)
    #print("IMG:"+str(image_tesseract_point_DM),file=fout)
    #dedisp_img_hi = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_hi),a_min=0,a_max=nsamps-1),axis=2)) 
    #dedisp_img_low = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_low),a_min=0,a_max=nsamps-1),axis=2))

    dedisp_timeseries_all = (((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_hi),a_min=0,a_max=nsamps-1),axis=2))*tdelays_frac) + 
                             ((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_low),a_min=0,a_max=nsamps-1),axis=2))*(1 - tdelays_frac))).sum(3)#.block_until_ready()

    #print("SHAPES:" + str(tdelays_idx_hi.shape),file=fout)
    #dedisp_timeseries_all = dedisp_img.sum(3)#.block_until_ready()
    
    dedisp_timeseries_all_cpu = (jax.device_put(dedisp_timeseries_all,jax.devices("cpu")[0]))
    #dedisp_img_cpu = (jax.device_put(dedisp_img,jax.devices("cpu")[0]))

    del DM_trials
    del freq_axis
    del tdelays
    del tdelays_idx_hi
    del tdelays_idx_low
    del tdelays_frac
    del idxs_all
    del corr_shifts_all_hi
    del corr_shifts_all_low
    del image_tesseract_point_DM
    #del dedisp_img_hi
    #del dedisp_img_low
    del dedisp_timeseries_all
    #del dedisp_img
    #jax.clear_caches()
    


    return dedisp_timeseries_all_cpu#,dedisp_img_cpu


#@jax.jit
def inner_dedisperse_keepfreqaxis_jit(image_tesseract_point,DM_trials_in,tsamp,freq_axis_in):#,fout):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #make JAX arrays
    #print(image_tesseract_point.shape)
    image_tesseract_point = image_tesseract_point#.numpy()
    DM_trials = jnp.array(DM_trials_in)
    freq_axis = jnp.array(freq_axis_in)

    #Delays
    gridsize = image_tesseract_point.shape[0]
    #print("SIZES:" + str(gridsize),file=fout)
    nchans = len(freq_axis)
    nsamps = image_tesseract_point.shape[-2]



    tdelays = -(((DM_trials[:,jnp.newaxis].repeat(nchans,axis=1))*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))).transpose())
    tdelays_idx_hi = jnp.array(jnp.ceil(tdelays/tsamp),dtype=jnp.int16)
    tdelays_idx_low = jnp.array(jnp.floor(tdelays/tsamp),dtype=jnp.int16)
    tdelays_frac = tdelays/tsamp - tdelays_idx_low

    #print("TDELHI_FREQ_DM:" + str(tdelays_idx_hi),file=fout)
    #print("TDELLOW_FREQ_DM:" + str(tdelays_idx_low),file=fout)
    #print("Trial DM: " + str(DM_trials.shape) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout)



    #rearrange shift idxs
    idxs_all = (np.arange(nsamps)[:,jnp.newaxis,jnp.newaxis]).repeat(nchans,axis=1).repeat(len(DM_trials),axis=2)
    corr_shifts_all_hi = np.clip(((-tdelays_idx_hi[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)#(idxs_all - shifts_all_hi)%nsamps
    corr_shifts_all_low = np.clip(((-tdelays_idx_low[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)#(idxs_all - shifts_all_low)%nsamps
    #print("TDEL_HI:" + str(corr_shifts_all_hi),file=fout)
    #print("TDEL_LOW:" + str(corr_shifts_all_low),file=fout)

    #apply delays
    tdelays_frac = tdelays_frac[jnp.newaxis,jnp.newaxis,jnp.newaxis,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1).repeat(nsamps,axis=2)
    corr_shifts_all_hi = corr_shifts_all_hi[jnp.newaxis,jnp.newaxis,:,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    corr_shifts_all_low = corr_shifts_all_low[jnp.newaxis,jnp.newaxis,:,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    image_tesseract_point_DM = image_tesseract_point[:,:,:,:,jnp.newaxis].repeat(len(DM_trials),axis=4)
    #print("IMG:"+str(image_tesseract_point_DM),file=fout)
    #dedisp_img_hi = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_hi),a_min=0,a_max=nsamps-1),axis=2))
    #dedisp_img_low = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_low),a_min=0,a_max=nsamps-1),axis=2))

    #dedisp_img = ((dedisp_img_hi*tdelays_frac) + (dedisp_img_low*(1 - tdelays_frac)))#.block_until_ready()

    dedisp_img = (((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_hi),a_min=0,a_max=nsamps-1),axis=2))*tdelays_frac) +
                  ((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_low),a_min=0,a_max=nsamps-1),axis=2))*(1 - tdelays_frac)))#.sum(3)#.block_until_ready()

    #print("SHAPES:" + str(tdelays_idx_hi.shape),file=fout)
    #dedisp_timeseries_all = dedisp_img.sum(3)#.block_until_ready()

    #dedisp_timeseries_all_cpu = (jax.device_put(dedisp_timeseries_all,jax.devices("cpu")[0]))
    dedisp_img_cpu = (jax.device_put(dedisp_img,jax.devices("cpu")[0]))

    del DM_trials
    del freq_axis
    del tdelays
    del tdelays_idx_hi
    del tdelays_idx_low
    del tdelays_frac
    del idxs_all
    del corr_shifts_all_hi
    del corr_shifts_all_low
    del image_tesseract_point_DM
    #del dedisp_img_hi
    #del dedisp_img_low
    #del dedisp_timeseries_all
    del dedisp_img
    #jax.clear_caches()



    return dedisp_img_cpu

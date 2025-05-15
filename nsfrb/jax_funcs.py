import jax
import jax.numpy as jnp
import numpy as np

"""
This file defines jit compiled functions to accelerate GPU compilation and computation.
"""




"""
matched filter + DM + SNR combined
"""
@jax.jit
def matched_filter_dedisp_snr_fft_jit(image_tesseract_input,PSFimg,corr_shifts_all,tdelays_frac,boxcar,noise,past_noise_N,noiseth):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #matched filter
    truensamps = boxcar.shape[3]
    
    gridsize_DEC,gridsize_RA = image_tesseract_input.shape[:2]
    nsamps = image_tesseract_input.shape[-2]


    #median subtraction
    print(truensamps,image_tesseract_input.shape)
    image_tesseract_point1 = image_tesseract_input[:,:,:nsamps-truensamps,:] - jnp.nanmedian(image_tesseract_input[:,:,:nsamps-truensamps,:],axis=2,keepdims=True)#(jnp.zeros_like(image_tesseract_input[:,:,:nsamps-truensamps,:] if nsamps==truensamps else jnp.nanmedian(image_tesseract_input[:,:,:nsamps-truensamps,:],axis=2,keepdims=True)))
    image_tesseract_point2 = image_tesseract_input[:,:,nsamps-truensamps:,:] - jnp.nanmedian(image_tesseract_input[:,:,nsamps-truensamps:,:],axis=2,keepdims=True)
    print(image_tesseract_point1.shape,image_tesseract_point2.shape)
    image_tesseract_point = jnp.concatenate([image_tesseract_point1,image_tesseract_point2],axis=2)

    #dedispersion
    print("HOWDY" + str(nsamps) + "  " + str(truensamps) + " " + str(image_tesseract_point.shape))
    nsamps = image_tesseract_point.shape[-2]
    nDM = tdelays_frac.shape[3]
    
    image_tesseract_filtered_dm = ((((jnp.take_along_axis(image_tesseract_point[:,:,:,jnp.newaxis,:].repeat(nDM,axis=3).repeat(2,axis=4),indices=corr_shifts_all,axis=2))*tdelays_frac).sum(4))[:,:,-truensamps:,:])

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
    print("FROM JAX:",image_tesseract_binned)
    mask = ((image_tesseract_binned - jnp.nanmedian(image_tesseract_binned,axis=4,keepdims=True) < noiseth*noise[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(nDM,3).repeat(truensamps,4)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    mask = mask.at[:].set((mask + ((noise==0)[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(nDM,3).repeat(truensamps,4)))>0)
    #compute noise and update
    noise = noise.at[:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned[:,:,:,0,:],axis=3,where=mask[:,:,:,0,:]
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))
    print("FROM JAX:",noise)
    
    #compute SNR
    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(jnp.sqrt(jnp.abs( ((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise[:,np.newaxis].repeat(nDM,1),(1,2)))))))[:,:,:,:,0].transpose(1,2,0,3)

    del mask
    del boxcar
    
    
    #mmatched filter
    gridsize_DEC,gridsize_RA = image_tesseract_binned_new.shape[:2]
    padby_DEC = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA = (gridsize_RA - PSFimg.shape[1])//2
    padby_DEC_img = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA_img = (gridsize_RA - PSFimg.shape[1])//2


    #image_tesseract_point
    del PSFimg
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),jax.device_put((image_tesseract_binned.argmax(4)).astype(jnp.uint8).transpose(1,2,0,3),jax.devices("cpu")[0])

"""
matched filter + DM + SNR combined
"""
@jax.jit
def matched_filter_dedisp_snr_fft_jit_no_append(image_tesseract_input,PSFimg,corr_shifts_all,tdelays_frac,boxcar,noise,past_noise_N,noiseth):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #matched filter
    truensamps = boxcar.shape[3]

    gridsize_DEC,gridsize_RA = image_tesseract_input.shape[:2]
    nsamps = image_tesseract_input.shape[-2]

    #median subtraction
    print(truensamps,image_tesseract_input.shape)
    image_tesseract_point = image_tesseract_input- jnp.nanmedian(image_tesseract_input,axis=2,keepdims=True)

    #dedispersion
    print("HOWDY" + str(nsamps) + "  " + str(truensamps) + " " + str(image_tesseract_point.shape))
    nsamps = image_tesseract_point.shape[-2]
    nDM = tdelays_frac.shape[3]

    image_tesseract_filtered_dm = ((((jnp.take_along_axis(image_tesseract_point[:,:,:,jnp.newaxis,:].repeat(nDM,axis=3).repeat(2,axis=4),indices=corr_shifts_all,axis=2))*tdelays_frac).sum(4)))

    del tdelays_frac
    del corr_shifts_all
    print("[NO APPEND] SHAPE AFTER DEDISPERSION:: ",image_tesseract_filtered_dm.shape)

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
    print(image_tesseract_binned)
    mask = ((image_tesseract_binned - jnp.nanmedian(image_tesseract_binned,axis=4,keepdims=True) < noiseth*noise[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(nDM,3).repeat(truensamps,4)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    mask = mask.at[:].set((mask + ((noise==0)[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(nDM,3).repeat(truensamps,4)))>0)
    #compute noise and update
    noise = noise.at[:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned[:,:,:,0,:],axis=3,where=mask[:,:,:,0,:]
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))

    #compute SNR
    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(jnp.sqrt(jnp.abs( ((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise[:,np.newaxis].repeat(nDM,1),(1,2)))))))[:,:,:,:,0].transpose(1,2,0,3)

    del mask
    del boxcar


    #mmatched filter
    gridsize_DEC,gridsize_RA = image_tesseract_binned_new.shape[:2]
    padby_DEC = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA = (gridsize_RA - PSFimg.shape[1])//2
    padby_DEC_img = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA_img = (gridsize_RA - PSFimg.shape[1])//2


    #image_tesseract_point
    del PSFimg
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),jax.device_put((image_tesseract_binned.argmax(4)).astype(jnp.uint8).transpose(1,2,0,3),jax.devices("cpu")[0])


"""
matched filter + SNR for data with no freq axis
"""
@jax.jit
def img_diff_jit_no_append(image_tesseract_input,PSFimg,boxcar,noise,past_noise_N,noiseth):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #matched filter
    gridsize_DEC,gridsize_RA = image_tesseract_input.shape[:2]
    truensamps = nsamps = image_tesseract_input.shape[2]

    #median subtraction
    print(truensamps,image_tesseract_input.shape)
    image_tesseract_point = (image_tesseract_input- jnp.nanmedian(image_tesseract_input,axis=2,keepdims=True))
    print("[IMG DIFF] SHAPE AFTER DEDISPERSION:: ",image_tesseract_point.shape)

    #boxcar filter
    image_tesseract_binned = jnp.nan_to_num(jnp.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(image_tesseract_point,
                                                            n=image_tesseract_point.shape[2],
                                                            axis=2,norm='backward')*jnp.fft.fft(boxcar,
                                                            n=image_tesseract_point.shape[2],axis=3,norm='backward'),
                                                        n=image_tesseract_point.shape[2],
                                                        axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)),
                                            nan=0,posinf=0,neginf=0)##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps

    del image_tesseract_point
    print(image_tesseract_binned)
    mask = ((image_tesseract_binned - jnp.nanmedian(image_tesseract_binned,axis=4,keepdims=True) < noiseth*noise[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(1,3).repeat(truensamps,4)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    mask = mask.at[:].set((mask + ((noise==0)[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(1,3).repeat(truensamps,4)))>0)
    #compute noise and update
    noise = noise.at[:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned[:,:,:,0,:],axis=3,where=mask[:,:,:,0,:]
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))

    #compute SNR
    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(jnp.sqrt(jnp.abs( ((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise[:,np.newaxis].repeat(1,1),(1,2)))))))[:,:,:,:,0].transpose(1,2,0,3)

    del mask
    del boxcar


    #mmatched filter
    gridsize_DEC,gridsize_RA = image_tesseract_binned_new.shape[:2]
    padby_DEC = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA = (gridsize_RA - PSFimg.shape[1])//2
    padby_DEC_img = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA_img = (gridsize_RA - PSFimg.shape[1])//2


    #image_tesseract_point
    del PSFimg
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),jax.device_put((image_tesseract_binned.argmax(4)).astype(jnp.uint8).transpose(1,2,0,3),jax.devices("cpu")[0])

"""
Brute-force Fast Folding Algorithm
"""
def ffa_jit(image_tesseract_input,PSFimg,boxcar,noise,past_noise_N,noiseth,idxs_full):
    """
    Brute-force Fast Folding Algorithm
    idxs_full -> (1,1,1,nsamps,nsamps,ntrialP)
    """
    #matched filter
    gridsize_DEC,gridsize_RA = image_tesseract_input.shape[:2]
    truensamps = nsamps = image_tesseract_input.shape[2]

    #median subtraction
    print(truensamps,image_tesseract_input.shape)
    image_tesseract_point = (image_tesseract_input- jnp.nanmedian(image_tesseract_input,axis=2,keepdims=True))
    print("[IMG DIFF] SHAPE AFTER DEDISPERSION:: ",image_tesseract_point.shape)

    #boxcar filter
    image_tesseract_binned = jnp.nan_to_num(jnp.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(image_tesseract_point,
                                                            n=image_tesseract_point.shape[2],
                                                            axis=2,norm='backward')*jnp.fft.fft(boxcar,
                                                            n=image_tesseract_point.shape[2],axis=3,norm='backward'),
                                                        n=image_tesseract_point.shape[2],
                                                        axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)),
                                            nan=0,posinf=0,neginf=0)##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps

    del image_tesseract_point
    print(image_tesseract_binned)
    mask = ((image_tesseract_binned - jnp.nanmedian(image_tesseract_binned,axis=4,keepdims=True) < noiseth*noise[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(1,3).repeat(truensamps,4)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    mask = mask.at[:].set((mask + ((noise==0)[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(1,3).repeat(truensamps,4)))>0)
    #compute noise and update
    noise = noise.at[:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned[:,:,:,0,:],axis=3,where=mask[:,:,:,0,:]
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))

    #fast folding
    image_tesseract_folded = image_tesseract_binned.at[:,:,:,0,:].set(np.nanmax((np.take_along_axis((image_tesseract_binned[:,:,:,0,:] - jnp.nanmedian(image_tesseract_binned*mask,axis=4))[:,:,:,:,np.newaxis,np.newaxis],idxs_full,axis=-3)).sum(-2,where=idxs_full!=0),-2)) #output should be nwidths x gridsize_DEC x gridsize_RA x nPs

    #compute SNR
    image_tesseract_binned_new = (image_tesseract_folded.at[:,:,:,:].set(jnp.sqrt(jnp.abs (image_tesseract_folded/jnp.expand_dims(noise[:,np.newaxis].repeat(1,1),(1,2)))))).transpose(1,2,0,3)

    del mask
    del boxcar
    
    
    #image_tesseract_point
    del PSFimg
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),jax.device_put((image_tesseract_binned.argmax(4)).astype(jnp.uint8).transpose(1,2,0,3),jax.devices("cpu")[0])


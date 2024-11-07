import jax
import jax.numpy as jnp
import numpy as np

"""
This file defines jit compiled functions to accelerate GPU compilation and computation.
"""





"""
no matched filter
"""
@jax.jit
def matched_filter_dedisp_snr_fft_jit_init(image_tesseract_point,corr_shifts_all,tdelays_frac,boxcar,noise,past_noise_N,noiseth):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """
    #dedispersion
    nsamps = image_tesseract_point.shape[-2]
    nDM = tdelays_frac.shape[3]
    truensamps = boxcar.shape[3]

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
    del boxcar

    #create masks
    mask = ((image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf

    #compute noise and update
    noise = noise.at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned,axis=4,where=mask
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))

    """
    noise = noise.at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned*mask,axis=4
                                                    )*jnp.sqrt(nsamps/(mask.sum(4))
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))
    """
    #compute SNR
    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise,(1,2)))))[:,:,:,:,0].transpose(1,2,0,3)


    del mask
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0])

"""
matched filter + DM + SNR combined
"""
@jax.jit
def matched_filter_dedisp_snr_fft_jit(image_tesseract_point,PSFimg,corr_shifts_all,tdelays_frac,boxcar,noise,past_noise_N,noiseth):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #matched filter
    truensamps = boxcar.shape[3]
    
    """gridsize_DEC,gridsize_RA = image_tesseract.shape[:2]
    gridsize_DEC*=2
    gridsize_RA*=2
    padby_DEC = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA = (gridsize_RA - PSFimg.shape[1])//2
    padby_DEC_img = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA_img = (gridsize_RA - PSFimg.shape[1])//2


    #image_tesseract_point 
    image_tesseract_point = jnp.real(
                                jnp.fft.fftshift(jnp.fft.ifft2(
                                    jnp.fft.fft2(jnp.pad(image_tesseract,
                                            ((padby_DEC_img,padby_DEC_img),(padby_RA_img,padby_RA_img),(0,0),(0,0))),
                                        axes=(0,1),s=(gridsize_DEC,gridsize_RA))*jnp.fft.fft2(jnp.pad(
                                            (PSFimg.repeat(image_tesseract.shape[2],axis=2)),
                                            ((padby_DEC,padby_DEC),(padby_RA,padby_RA),(0,0),(0,0))),
                                            axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                    ,axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                ,axes=(0,1)))[gridsize_DEC//4:(gridsize_DEC//4) + gridsize_DEC//2,gridsize_RA//4:(gridsize_RA//4) + gridsize_RA//2,:,:]
    """
    gridsize_DEC,gridsize_RA = image_tesseract_point.shape[:2]
    
    #del image_tesseract
    #del PSFimg
    #del image_tesseract

    #dedispersion
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
    #del boxcar

    #create masks
    #mask = ((image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    #mask = ((image_tesseract_binned < noiseth*noise[:,np.newaxis,np.newaxis,:,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(truensamps,4)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    mask = ((image_tesseract_binned - jnp.nanmedian(image_tesseract_binned,axis=4,keepdims=True) < noiseth*noise[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis].repeat(gridsize_DEC,1).repeat(gridsize_RA,2).repeat(nDM,3).repeat(truensamps,4)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    #compute noise and update
    noise = noise.at[:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned[:,:,:,0,:],axis=3,where=mask[:,:,:,0,:]
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))
    
    """
    noise = noise.at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned*mask,axis=4
                                                    )*jnp.sqrt(nsamps/(mask.sum(4))
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))
    """
    #compute SNR
    #image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(jnp.sqrt(jnp.abs( ((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise[:,0:1].repeat(nDM,1),(1,2)))))))[:,:,:,:,0].transpose(1,2,0,3)
    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(jnp.sqrt(jnp.abs( ((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise[:,np.newaxis].repeat(nDM,1),(1,2)))))))[:,:,:,:,0].transpose(1,2,0,3)
    #image_tesseract_TOAs = image_tesseract_binned.at[:,:,:,:,1].set(image_tesseract_binned.argmax(4)).astype(jnp.uint8).transpose(1,2,0,3)

    del mask
    del boxcar
    
    
    #mmatched filter
    gridsize_DEC,gridsize_RA = image_tesseract_binned_new.shape[:2]
    gridsize_DEC*=2
    gridsize_RA*=2
    padby_DEC = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA = (gridsize_RA - PSFimg.shape[1])//2
    padby_DEC_img = (gridsize_DEC - PSFimg.shape[0])//2
    padby_RA_img = (gridsize_RA - PSFimg.shape[1])//2


    #image_tesseract_point
    image_tesseract_final = jnp.real(
                                jnp.fft.fftshift(jnp.fft.ifft2(
                                    jnp.fft.fft2(jnp.pad(image_tesseract_binned_new,
                                            ((padby_DEC_img,padby_DEC_img),(padby_RA_img,padby_RA_img),(0,0),(0,0))),
                                        axes=(0,1),s=(gridsize_DEC,gridsize_RA))*jnp.fft.fft2(jnp.pad(
                                            (PSFimg.repeat(image_tesseract_binned_new.shape[2],axis=2).repeat(image_tesseract_binned_new.shape[3],axis=3)),
                                            ((padby_DEC,padby_DEC),(padby_RA,padby_RA),(0,0),(0,0))),
                                            axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                    ,axes=(0,1),s=(gridsize_DEC,gridsize_RA))
                                ,axes=(0,1)))[gridsize_DEC//4:(gridsize_DEC//4) + gridsize_DEC//2,gridsize_RA//4:(gridsize_RA//4) + gridsize_RA//2,:,:]
    
    del PSFimg
    return jax.device_put(image_tesseract_final,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),jax.device_put((image_tesseract_binned.argmax(4)).astype(jnp.uint8).transpose(1,2,0,3),jax.devices("cpu")[0])

"""
matched filter
"""
@jax.jit
def matched_filter_fft_jit(image_tesseract,PSFimg):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """
    gridsize = image_tesseract.shape[0]
    padby = (gridsize - PSFimg.shape[0])//2

    return jax.device_put(jnp.real(jnp.fft.fftshift(
                                                jnp.fft.ifft2(
                                                    jnp.fft.fft2(jnp.fft.ifftshift(image_tesseract,axes=(0,1)),
                                                        axes=(0,1),s=(gridsize,gridsize))*jnp.fft.fft2(jnp.pad(
                                                            (jnp.fft.ifftshift(PSFimg.repeat(image_tesseract.shape[2],axis=2),axes=(0,1))),
                                                            ((padby,padby),(padby,padby),(0,0),(0,0))),
                                                            axes=(0,1),s=(gridsize,gridsize))
                                                    ,axes=(0,1),s=(gridsize,gridsize))
                                                ,axes=(0,1))),jax.devices("cpu")[0])



    #return jax.device_put(jnp.real(jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fft2(image_tesseract,axes=(0,1),s=image_tesseract.shape[:2])*jnp.fft.fft2(PSFimg,axes=(0,1),s=image_tesseract.shape[:2]),axes=(0,1),s=image_tesseract.shape[:2]),axes=(0,1))),jax.devices("cpu")[0])

"""
combined function
"""
@jax.jit
def dedisp_snr_fft_jit_0(image_tesseract_point,corr_shifts_all,tdelays_frac,boxcar,noise,past_noise_N,noiseth,i,j):#,fout):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #make JAX arrays 
    #print(image_tesseract_point.shape) 
    #image_tesseract_point = image_tesseract_point#.numpy()
    """
    DM_trials = jax.device_put(jnp.array(DM_trials_in),jax.devices()[0])
    freq_axis = jax.device_put(jnp.array(freq_axis_in),jax.devices()[0])
    """

    #Delays
    gridsize = image_tesseract_point.shape[0]
    #print("SIZES:" + str(gridsize),file=fout)
    #nchans = len(freq_axis)
    nsamps = image_tesseract_point.shape[-2]
    truensamps = boxcar.shape[3]
    nDM = tdelays_frac.shape[3]
    """
    #tdelays = -(((DM_trials[:,jnp.newaxis].repeat(nchans,axis=1))*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))).transpose())
    
    tdelaysall = jnp.zeros((nDM,nchans*2),dtype=jnp.int16) #jnp.device_put(jnp.zeros((len(DM_trials),nchans*2),dtype=jnp.int16),jax.devices()[0])
    tdelaysall = tdelaysall.at[:,1::2].set(jnp.array(jnp.ceil(tdelays/tsamp),dtype=jnp.int16))
    tdelaysall = tdelaysall.at[:,0::2].set(jnp.array(jnp.floor(tdelays/tsamp),dtype=jnp.int16))
    tdelays_frac = jnp.concatenate([tdelays/tsamp - tdelaysall[:,0::2],1 - (tdelays/tsamp - tdelaysall[:,0::2])],axis=1)

    #rearrange shift idxs and expand axes
    idxs_all = (np.arange(nsamps)[:,jnp.newaxis,jnp.newaxis]).repeat(nDM,axis=1).repeat(2*nchans,axis=2)
    corr_shifts_all = jnp.clip(((-tdelaysall[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)[jnp.newaxis,jnp.newaxis,:truensamps,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    tdelays_frac = tdelays_frac[jnp.newaxis,jnp.newaxis,jnp.newaxis,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1).repeat(truensamps,axis=2)
    """
    #image_tesseract_point_DM = image_tesseract_point[:,:,:,jnp.newaxis,:].repeat(nDM,axis=3).repeat(2,axis=4)
    
    #apply delays
    image_tesseract_filtered_dm = ((jnp.take_along_axis(image_tesseract_point[:,:,:,jnp.newaxis,:].repeat(nDM,axis=3).repeat(2,axis=4),indices=corr_shifts_all,axis=2))*tdelays_frac).sum(4) 

    #del tdelays
    #del tdelaysall
    del tdelays_frac
    #del idxs_all
    del corr_shifts_all
    #del image_tesseract_point_DM
    """
    tdelays_idx_hi = jnp.array(jnp.ceil(tdelays/tsamp),dtype=jnp.int16)
    tdelays_idx_low = jnp.array(jnp.floor(tdelays/tsamp),dtype=jnp.int16)
    tdelays_frac = tdelays/tsamp - tdelays_idx_low

    #print("TDELHI_FREQ_DM:" + str(tdelays_idx_hi),file=fout)
    #print("TDELLOW_FREQ_DM:" + str(tdelays_idx_low),file=fout)
    #print("Trial DM: " + str(DM_trials.shape) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout)

    #rearrange shift idxs
    idxs_all = (np.arange(nsamps)[:,jnp.newaxis,jnp.newaxis]).repeatnchans,axis=1).repeat(len(DM_trials),axis=2)
    corr_shifts_all_hi = np.clip(((-tdelays_idx_hi[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)[:truensamps]#(idxs_all - shifts_all_hi)%nsamps
    corr_shifts_all_low = np.clip(((-tdelays_idx_low[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)[:truensamps]#(idxs_all - shifts_all_low)%nsamps
    #print("TDEL_HI:" + str(corr_shifts_all_hi),file=fout)
    #print("TDEL_LOW:" + str(corr_shifts_all_low),file=fout)

    #apply delays
    tdelays_frac = tdelays_frac[jnp.newaxis,jnp.newaxis,jnp.newaxis,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1).repeat(truensamps,axis=2)
    corr_shifts_all_hi = corr_shifts_all_hi[jnp.newaxis,jnp.newaxis,:,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    corr_shifts_all_low = corr_shifts_all_low[jnp.newaxis,jnp.newaxis,:,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    image_tesseract_point_DM = image_tesseract_point[:,:,:,:,jnp.newaxis].repeat(len(DM_trials),axis=4)
    #print("IMG:"+str(image_tesseract_point_DM),file=fout)
    #dedisp_img_hi = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_hi),a_min=0,a_max=nsamps-1),axis=2))
    #dedisp_img_low = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_low),a_min=0,a_max=nsamps-1),axis=2))

    image_tesseract_filtered_dm = ((((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.array(corr_shifts_all_hi),axis=2))*tdelays_frac) + ((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.array(corr_shifts_all_low),axis=2))*(1 - tdelays_frac))).sum(3))#.block_until_ready()


    #print("SHAPES:" + str(tdelays_idx_hi.shape),file=fout)
    #dedisp_timeseries_all = dedisp_img.sum(3)#.block_until_ready()

    #dedisp_timeseries_all_cpu = (jax.device_put(dedisp_timeseries_all,jax.devices("cpu")[0]))
    #dedisp_img_cpu = (jax.device_put(dedisp_img,jax.devices("cpu")[0]))

    del DM_trials
    #del freq_axis
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
    #del dedisp_img
    #jax.clear_caches()

    #return dedisp_timeseries_all_cpu,i,j#,dedisp_img_cpu
    """

    #####NOW BOXCAR FILTER
    #take fourier transform of boxcar and image
    """
    image_tesseract_binned = jnp.nan_to_num(jnp.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(image_tesseract_filtered_dm,n=image_tesseract_filtered_dm.shape[2],axis=2,norm='backward')*jnp.fft.fft(jax.device_put(jnp.array(boxcar),jax.devices()[0]),n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),
                                                                                                                                            n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)) ,nan=0,posinf=0,neginf=0)##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps
    """
    image_tesseract_binned = jnp.nan_to_num(jnp.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(image_tesseract_filtered_dm,n=image_tesseract_filtered_dm.shape[2],axis=2,norm='backward')*jnp.fft.fft(boxcar,n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),
                                                                                                                                            n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)) ,nan=0,posinf=0,neginf=0)##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps

    del image_tesseract_filtered_dm
    del boxcar

    #create masks
    mask = ((image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    #image_tesseract_binned.at[:,:,:,:,:].set(jnp.nan_to_num(image_tesseract_binned,nan=0,posinf=0,neginf=0))
    #mask.at[:,:,:,:,:].set(mask*(image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True))) #less than noise threshold
    #nw,subgridsize_DEC,subgridsize_RA,ndms,nsamps = image_tesseract_binned.shape#[4]

    #compute noise and update
    #noise = jnp.array(noise)
    """
    noise = jax.device_put(jnp.array(noise),jax.devices()[0]).at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned*mask,axis=4
                                                    )*jnp.sqrt(nsamps/(mask.sum(4))
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))
    """
    noise = noise.at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned*mask,axis=4
                                                    )*jnp.sqrt(nsamps/(mask.sum(4))
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))
    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise,(1,2)))))[:,:,:,:,0].transpose(1,2,0,3)#0,2).transpose(0,1))

    #imgout[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j)*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = imgout.at[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j)*subgridsize_RA:(j+1)*subgridsize_RA,:,:].set(jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]))



    del mask
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),i,j




@jax.jit
def dedisp_snr_fft_jit_1(image_tesseract_point,DM_trials_in,tsamp,freq_axis_in,boxcar,noise,past_noise_N,noiseth,i,j):#,fout):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #make JAX arrays 
    #print(image_tesseract_point.shape) 
    #image_tesseract_point = image_tesseract_point#.numpy()
    DM_trials = jax.device_put(jnp.array(DM_trials_in),jax.devices()[1])
    freq_axis = jax.device_put(jnp.array(freq_axis_in),jax.devices()[1])

    #Delays
    gridsize = image_tesseract_point.shape[0]
    #print("SIZES:" + str(gridsize),file=fout)
    nchans = len(freq_axis)
    nsamps = image_tesseract_point.shape[-2]
    truensamps = boxcar.shape[3]

    tdelays = -(((DM_trials[:,jnp.newaxis].repeat(nchans,axis=1))*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))).transpose())

    tdelaysall = jnp.device_put(jnp.zeros((len(DM_trials),nchans*2),dtype=jnp.int16),jax.devices()[1])
    tdelaysall = tdelaysall.at[:,1::2].set(jnp.array(jnp.ceil(tdelays/tsamp),dtype=jnp.int16))
    tdelaysall = tdelaysall.at[:,0::2].set(jnp.array(jnp.floor(tdelays/tsamp),dtype=jnp.int16))
    tdelays_frac = jnp.concatenate([tdelays/tsamp - tdelaysall[:,0::2],1 - (tdelays/tsamp - tdelaysall[:,0::2])],axis=1)

    #rearrange shift idxs and expand axes
    idxs_all = (np.arange(nsamps)[:,jnp.newaxis,jnp.newaxis]).repeat(len(DM_trials),axis=1).repeat(2*nchans,axis=2)
    corr_shifts_all = jnp.clip(((-tdelaysall[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)[jnp.newaxis,jnp.newaxis,:truensamps,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    tdelays_frac = tdelays_frac[jnp.newaxis,jnp.newaxis,jnp.newaxis,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1).repeat(truensamps,axis=2)
    image_tesseract_point_DM = image_tesseract_point[:,:,:,jnp.newaxis,:].repeat(len(DM_trials),axis=3).repeat(2,axis=4)

    #apply delays
    image_tesseract_filtered_dm = ((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.array(corr_shifts_all),axis=2))*tdelays_frac).sum(4)

    del DM_trials
    del freq_axis
    del tdelays
    del tdelaysall
    del tdelays_frac
    del idxs_all
    del corr_shifts_all
    del image_tesseract_point_DM


    """
    tdelays_idx_hi = jnp.array(jnp.ceil(tdelays/tsamp),dtype=jnp.int16)
    tdelays_idx_low = jnp.array(jnp.floor(tdelays/tsamp),dtype=jnp.int16)
    tdelays_frac = tdelays/tsamp - tdelays_idx_low

    #print("TDELHI_FREQ_DM:" + str(tdelays_idx_hi),file=fout)
    #print("TDELLOW_FREQ_DM:" + str(tdelays_idx_low),file=fout)
    #print("Trial DM: " + str(DM_trials.shape) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout)

    #rearrange shift idxs
    idxs_all = (np.arange(nsamps)[:,jnp.newaxis,jnp.newaxis]).repeat(nchans,axis=1).repeat(len(DM_trials),axis=2)
    corr_shifts_all_hi = np.clip(((-tdelays_idx_hi[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)[:truensamps]#(idxs_all - shifts_all_hi)%nsamps
    corr_shifts_all_low = np.clip(((-tdelays_idx_low[jnp.newaxis,:,:].repeat(nsamps,axis=0) + idxs_all))%nsamps,a_min=0,a_max=nsamps-1)[:truensamps]#(idxs_all - shifts_all_low)%nsamps
    #print("TDEL_HI:" + str(corr_shifts_all_hi),file=fout)
    #print("TDEL_LOW:" + str(corr_shifts_all_low),file=fout)


    #apply delays
    tdelays_frac = tdelays_frac[jnp.newaxis,jnp.newaxis,jnp.newaxis,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1).repeat(truensamps,axis=2)
    corr_shifts_all_hi = corr_shifts_all_hi[jnp.newaxis,jnp.newaxis,:,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    corr_shifts_all_low = corr_shifts_all_low[jnp.newaxis,jnp.newaxis,:,:,:].repeat(image_tesseract_point.shape[0],axis=0).repeat(image_tesseract_point.shape[1],axis=1)
    image_tesseract_point_DM = image_tesseract_point[:,:,:,:,jnp.newaxis].repeat(len(DM_trials),axis=4)
    #print("IMG:"+str(image_tesseract_point_DM),file=fout)
    #dedisp_img_hi = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_hi),a_min=0,a_max=nsamps-1),axis=2))
    #dedisp_img_low = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_low),a_min=0,a_max=nsamps-1),axis=2))

    image_tesseract_filtered_dm = ((((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.array(corr_shifts_all_hi),axis=2))*tdelays_frac) + ((jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.array(corr_shifts_all_low),axis=2))*(1 - tdelays_frac))).sum(3))#.block_until_ready()


    #print("SHAPES:" + str(tdelays_idx_hi.shape),file=fout)
    #dedisp_timeseries_all = dedisp_img.sum(3)#.block_until_ready()

    #dedisp_timeseries_all_cpu = (jax.device_put(dedisp_timeseries_all,jax.devices("cpu")[0]))
    #dedisp_img_cpu = (jax.device_put(dedisp_img,jax.devices("cpu")[0]))

    del DM_trials
    #del freq_axis
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
    #del dedisp_img
    #jax.clear_caches()
    


    #return dedisp_timeseries_all_cpu,i,j#,dedisp_img_cpu
    """

    #####NOW BOXCAR FILTER
    #take fourier transform of boxcar and image
    image_tesseract_binned = jnp.nan_to_num(jnp.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(image_tesseract_filtered_dm,n=image_tesseract_filtered_dm.shape[2],axis=2,norm='backward')*jnp.fft.fft(jax.device_put(jnp.array(boxcar),jax.devices()[1]),n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),
                                                                                                                                            n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)) ,nan=0,posinf=0,neginf=0)##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps


    del image_tesseract_filtered_dm
    del boxcar

    #create masks
    mask = ((image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    #image_tesseract_binned.at[:,:,:,:,:].set(jnp.nan_to_num(image_tesseract_binned,nan=0,posinf=0,neginf=0))
    #mask.at[:,:,:,:,:].set(mask*(image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True))) #less than noise threshold
    nw,subgridsize_DEC,subgridsize_RA,ndms,nsamps = image_tesseract_binned.shape#[4]

    #compute noise and update
    #noise = jnp.array(noise)
    noise = jax.device_put(jnp.array(noise),jax.devices()[1]).at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned*mask,axis=4
                                                    )*jnp.sqrt(nsamps/(mask.sum(4))
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))

    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise,(1,2)))))[:,:,:,:,0].transpose(1,2,0,3)#0,2).transpose(0,1))

    #imgout[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j)*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = imgout.at[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j)*subgridsize_RA:(j+1)*subgridsize_RA,:,:].set(jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]))



    del mask
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),i,j



"""
isolated functions
"""
@jax.jit
def inner_snr_fft_jit_1(image_tesseract_filtered_dm,boxcar,noise,past_noise_N,noiseth,i,j):
    """
    This function replaces pyptorch with JAX so that JIT computation can be invoked
    """



    #take fourier transform of boxcar and image
    image_tesseract_binned = jnp.nan_to_num(jnp.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(jax.device_put(jnp.array(image_tesseract_filtered_dm),jax.devices()[1]),n=image_tesseract_filtered_dm.shape[2],axis=2,norm='backward')*jnp.fft.fft(jnp.array(boxcar),n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),
                                                                                                                                            n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)) ,nan=0,posinf=0,neginf=0)##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps


    del image_tesseract_filtered_dm
    del boxcar

    #create masks
    mask = ((image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    #image_tesseract_binned.at[:,:,:,:,:].set(jnp.nan_to_num(image_tesseract_binned,nan=0,posinf=0,neginf=0))
    #mask.at[:,:,:,:,:].set(mask*(image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True))) #less than noise threshold
    nw,subgridsize_DEC,subgridsize_RA,ndms,nsamps = image_tesseract_binned.shape#[4]

    #compute noise and update
    #noise = jnp.array(noise)
    noise = jax.device_put(jnp.array(noise),jax.devices()[1]).at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned*mask,axis=4
                                                    )*jnp.sqrt(nsamps/(mask.sum(4))
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))

    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise,(1,2)))))[:,:,:,:,0].transpose(1,2,0,3)#0,2).transpose(0,1))

    #imgout[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j)*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = imgout.at[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j)*subgridsize_RA:(j+1)*subgridsize_RA,:,:].set(jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]))



    del mask
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),i,j 




@jax.jit
def inner_snr_fft_jit_0(image_tesseract_filtered_dm,boxcar,noise,past_noise_N,noiseth,i,j):
    """
    This function replaces pyptorch with JAX so that JIT computation can be invoked
    """
   


    #take fourier transform of boxcar and image
    image_tesseract_binned = jnp.nan_to_num(jnp.real(jnp.fft.ifftshift(
                                            jnp.fft.ifft(
                                                jnp.fft.fft(jax.device_put(jnp.array(image_tesseract_filtered_dm),jax.devices()[0]),n=image_tesseract_filtered_dm.shape[2],axis=2,norm='backward')*jnp.fft.fft(jnp.array(boxcar),n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),
                                                                                                                                            n=image_tesseract_filtered_dm.shape[2],axis=3,norm='backward'),axes=3)).transpose((0,1,2,4,3)) ,nan=0,posinf=0,neginf=0)##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps
    
                                            
    del image_tesseract_filtered_dm
    del boxcar

    #create masks
    mask = ((image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True)))*jnp.logical_not(jnp.logical_or(jnp.isinf(image_tesseract_binned),jnp.isnan(image_tesseract_binned))) #not nan or inf
    #image_tesseract_binned.at[:,:,:,:,:].set(jnp.nan_to_num(image_tesseract_binned,nan=0,posinf=0,neginf=0))
    #mask.at[:,:,:,:,:].set(mask*(image_tesseract_binned < jnp.nanquantile(jnp.nanmax(image_tesseract_binned,axis=4,keepdims=True),noiseth,axis=(1,2),keepdims=True))) #less than noise threshold
    nw,subgridsize_DEC,subgridsize_RA,ndms,nsamps = image_tesseract_binned.shape#[4]
    
    #compute noise and update
    #noise = jnp.array(noise)
    noise = jax.device_put(jnp.array(noise),jax.devices()[0]).at[:,:].set(((jnp.array(noise*past_noise_N)) + ((jnp.nanmedian(
                                            jnp.nanmedian(
                                                jnp.nanstd(
                                                    image_tesseract_binned*mask,axis=4
                                                    )*jnp.sqrt(nsamps/(mask.sum(4))
                                                ),axis=1
                                            ),axis=1
                                        ))))/(past_noise_N+1))
    

    image_tesseract_binned_new = (image_tesseract_binned.at[:,:,:,:,0].set(((image_tesseract_binned.max(4) - jnp.nanmedian(image_tesseract_binned*mask,axis=4))/jnp.expand_dims(noise,(1,2)))))[:,:,:,:,0].transpose(1,2,0,3)#0,2).transpose(0,1))
    
    #imgout[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j)*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = imgout.at[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j)*subgridsize_RA:(j+1)*subgridsize_RA,:,:].set(jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]))
    
    del mask
    return jax.device_put(image_tesseract_binned_new,jax.devices("cpu")[0]),jax.device_put(noise,jax.devices("cpu")[0]),i,j


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
def inner_dedisperse_jit_1(image_tesseract_point,DM_trials_in,tsamp,freq_axis_in,i,j):#,fout):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #make JAX arrays 
    #print(image_tesseract_point.shape) 
    #image_tesseract_point = image_tesseract_point#.numpy()
    DM_trials = jax.device_put(jnp.array(DM_trials_in),jax.devices()[1])
    freq_axis = jax.device_put(jnp.array(freq_axis_in),jax.devices()[1])

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



    return dedisp_timeseries_all_cpu,i,j#,dedisp_img_cpu

@jax.jit
def inner_dedisperse_jit_0(image_tesseract_point,DM_trials_in,tsamp,freq_axis_in,i,j):#,fout):
    """
    This function replaces pytorch with JAX so that JIT computation can be invoked
    """

    #make JAX arrays 
    #print(image_tesseract_point.shape) 
    #image_tesseract_point = image_tesseract_point#.numpy()
    DM_trials = jax.device_put(jnp.array(DM_trials_in),jax.devices()[0])
    freq_axis = jax.device_put(jnp.array(freq_axis_in),jax.devices()[0])

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
    


    return dedisp_timeseries_all_cpu,i,j#,dedisp_img_cpu


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

import jax
import jax.numpy as jnp
import numpy as np

"""
This file defines jit compiled functions to accelerate GPU compilation and computation.
"""

@jax.jit
def inner_dedisperse_jit(image_tesseract_point,DM_trials_in,tsamp,freq_axis_in):#,fout):
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
    dedisp_img_hi = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_hi),a_min=0,a_max=nsamps-1),axis=2)) 
    dedisp_img_low = (jnp.take_along_axis(jnp.array(image_tesseract_point_DM),indices=jnp.clip(jnp.array(corr_shifts_all_low),a_min=0,a_max=nsamps-1),axis=2))

    dedisp_img = ((dedisp_img_hi*tdelays_frac) + (dedisp_img_low*(1 - tdelays_frac)))#.block_until_ready()

    #print("SHAPES:" + str(tdelays_idx_hi.shape),file=fout)
    dedisp_timeseries_all = dedisp_img.sum(3)#.block_until_ready()
    
    dedisp_timeseries_all_cpu = (jax.device_put(dedisp_timeseries_all,jax.devices("cpu")[0]))
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
    del dedisp_img_hi
    del dedisp_img_low
    del dedisp_timeseries_all
    del dedisp_img
    #jax.clear_caches()
    


    return dedisp_timeseries_all_cpu,dedisp_img_cpu

"""
#@jax.jit
def inner_dedisperse_jit(image_tesseract_point,DM_trials,tsamp,freq_axis,device,fout):
    #make cuda tensors
    print(torch.cuda.is_available())
    print(image_tesseract_point.shape)
    freq_axis = torch.from_numpy(freq_axis).to(device)
    DM_trials = torch.from_numpy(DM_trials).to(device)

    #Delays
    gridsize = image_tesseract_point.shape[0]
    print("SIZES:" + str(gridsize),file=fout) 
    nchans = len(freq_axis)
    nsamps = image_tesseract_point.shape[-2]
    tdelays = -(((DM_trials.unsqueeze(1).expand(-1,nchans))*4.15*(((torch.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))).transpose(0,1))
    tdelays_idx_hi = torch.ceil(tdelays/tsamp).int()
    tdelays_idx_low = torch.floor(tdelays/tsamp).int()
    tdelays_frac = tdelays/tsamp - tdelays_idx_low
    print("TDELHI_FREQ_DM:" + str(tdelays_idx_hi),file=fout)
    print("TDELLOW_FREQ_DM:" + str(tdelays_idx_low),file=fout)
    print("Trial DM: " + str(DM_trials.shape) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout)
    torch.cuda.empty_cache()


    #rearrange shift idxs
    idxs_all = (torch.arange(nsamps).unsqueeze(1).unsqueeze(1)).expand(-1,nchans,len(DM_trials)).to(device)
    corr_shifts_all_hi = torch.clip(((-tdelays_idx_hi.unsqueeze(0).expand(nsamps,-1,-1) + idxs_all))%nsamps,min=0,max=nsamps-1)#(idxs_all - shifts_all_hi)%nsamps
    corr_shifts_all_low = torch.clip(((-tdelays_idx_low.unsqueeze(0).expand(nsamps,-1,-1) + idxs_all))%nsamps,min=0,max=nsamps-1)#(idxs_all - shifts_all_low)%nsamps
    print("TDEL_HI:" + str(corr_shifts_all_hi),file=fout)
    print("TDEL_LOW:" + str(corr_shifts_all_low),file=fout)    

    #apply delays
    tdelays_frac = tdelays_frac.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],nsamps,-1,-1)
    corr_shifts_all_hi = corr_shifts_all_hi.long().unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,-1,-1)
    corr_shifts_all_low = corr_shifts_all_low.long().unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,-1,-1)
    image_tesseract_point_DM = image_tesseract_point.unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))
    print("IMG:"+str(image_tesseract_point_DM),file=fout)
    dedisp_img_hi = (torch.gather(image_tesseract_point_DM.double().to("cpu"),dim=2,index=torch.clip(corr_shifts_all_hi.to("cpu"),min=0,max=nsamps-1)))
    dedisp_img_low = (torch.gather(image_tesseract_point_DM.double().to("cpu"),dim=2,index=torch.clip(corr_shifts_all_low.to("cpu"),min=0,max=nsamps-1)))

    dedisp_img = ((dedisp_img_hi.to(device)*tdelays_frac.double().to(device)) + (dedisp_img_low.to(device)*(1 - tdelays_frac.double().to(device)))).to("cpu")

    print("SHAPES:" + str(tdelays_idx_hi.to("cpu").shape),file=fout)
    dedisp_timeseries_all = (dedisp_img.sum(3)).to("cpu").float()
    torch.cuda.empty_cache()

    return dedisp_timeseries_all,dedisp_img
"""

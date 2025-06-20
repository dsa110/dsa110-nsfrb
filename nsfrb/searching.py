import numpy as np
import glob
import jax
import jax.numpy as jnp
import sys
from matplotlib import pyplot as plt
import random
import copy
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import truncnorm
from scipy.signal import peak_widths
from scipy.stats import norm
import os
from PIL import Image,ImageOps
#from gen_dmtrials_copy import gen_dm
import time
import torch
from torch.nn import functional as tf
from nsfrb import config
from scipy.interpolate import interp1d
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from nsfrb import simulating as sim
from simulations_and_classifications import generate_PSF_images as scPSF
from nsfrb.outputlogging import printlog,numpy_to_fits
from nsfrb.imaging import uv_to_pix
from nsfrb.planning import get_RA_cutoff
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
#from pytorch_dedispersion import dedispersion,boxcar_filter,candidate_finder
from astropy.time import Time
from nsfrb.config import NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,NSFRB_CANDDADA_SLOW_KEY,NSFRB_SRCHDADA_SLOW_KEY,NSFRB_TOADADA_SLOW_KEY,NSFRB_CANDDADA_IMGDIFF_KEY,NSFRB_SRCHDADA_IMGDIFF_KEY,NSFRB_TOADADA_IMGDIFF_KEY
#from realtime.rtwriter import rtwrite

fsize=45
fsize2=35
plt.rcParams.update({
                    'font.size': fsize,
                    'font.family': 'sans-serif',
                    'axes.labelsize': fsize,
                    'axes.titlesize': fsize,
                    'xtick.labelsize': fsize,
                    'ytick.labelsize': fsize,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 1,
                    'lines.markersize': 5,
                    'legend.fontsize': fsize2,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})

"""
This file contains Python functions for the offline slow transient (NSFRB) search. 
Myles Sherman
"""

"""
Directory for output data
"""
#output_dir = "/media/ubuntu/ssd/sherman/NSFRB_search_output/"
#output_dir = "./NSFRB_search_output/"
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
#cwd = os.environ['NSFRBDIR']
#sys.path.append(cwd + "/") 
from nsfrb.config import *
from nsfrb.noise import noise_update,noise_dir,noise_update_all
from nsfrb import jax_funcs

from nsfrb.config import cwd,frame_dir,psf_dir,noise_dir,output_file,timelogfile,processfile,error_file,freq_axis,srchtx_file,srchtime_file,minDM,maxDM,run_file

try:
    from nsfrb.config import cand_dir
except Exception as exc:
    printlog(exc,output_file=error_file)

try:
    torch.multiprocessing.set_start_method("spawn")
except:
    printlog("Failed to set torch multiprocess method...sucks for you",output_file=error_file)


"""
from dask.distributed import Client,Queue,fire_and_forget
#if the dask scheduler is set up, put the cand file name in the queue
QSETUP = False
if 'DASKPORT' in os.environ.keys():
    try:
        QCLIENT = Client("tcp://127.0.0.1:"+os.environ['DASKPORT'],timeout=1)#get_client()
        QSETUP = True
        QQUEUE = Queue("cand_cutter_queue")
    except TimeoutError as exc:
        printlog("Scheduler not started, cannot send to queue",output_file=processfile)
    except OSError as exc:
        printlog("Scheduler not started, cannot send to queue",output_file=processfile)
"""
#initialize jax device so we alternate between them if only 1 batch
jaxdev = 0
jax_inuse = [False,False]


#create axes
RA_axis,DEC_axis,elev = uv_to_pix(Time.now().mjd,gridsize,pixperFWHM=config.pixperFWHM)
print(DEC_axis)
time_axis = np.linspace(0,T,nsamps) #ms

#DM trials
def gen_dm(dm1,dm2,tol,nu,nchan,tsamp,B,nsamps,ZERO=True):
    #tol = 1.25 # S/N loss tolerance
    #nu = 1.405 # center frequency (GHz)
    #nchan = 1024 # number of channels
    #tsamp = 262.144 # sampling time (microseconds)
    #B = 250./nchan # bandwidth per channel (MHz)

    ndms = 1
    dm_prev = dm1
    dm = 0.
    dms = []
    while dm<dm2:

        n2 = nchan**2.
        alp = 1./(16.+n2)
        bet = tsamp**2.
        dm = n2*alp*dm_prev + np.sqrt(16.*alp*(tol**2.-n2*alp)*dm_prev**2.+16.*alp*bet*(tol**2.-1.)*(nu**3./8.3/B)**2.)
        dm_prev = dm
        ndms += 1
        #print(dm)
        dms.append(dm)
    dms = np.array(dms)
    #limit maximum DM using the number of samples
    fmin=nu - (B*nchan*1e-3/2) #GHz
    fmax=nu + (B*nchan*1e-3/2)
    tdms = np.ceil((4.15)*np.array(dms)*((1/fmin)**2 - (1/fmax)**2)/tsamp) #samps
    dms = dms[tdms<nsamps]

    #print('DM trials:',ndms)
    if ZERO: return [0] +list(dms)
    else: return dms

def gen_dm_shifts(DM_trials,freq_axis,tsamp,nsamps,gridsize=1,outputwraps=False,maxshift=None): #note, you shouldn't need to set gridsize
    nDM = len(DM_trials)
    nchans = len(freq_axis)
    fmin =np.nanmin(freq_axis)
    fmax = np.nanmax(freq_axis)

    tdelays = -(((DM_trials[:,np.newaxis].repeat(nchans,axis=1))*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))).transpose())

    tdelaysall = np.zeros((nchans*2,nDM),dtype=np.int16) #jnp.device_put(jnp.zeros((len(DM_trials),nchans*2),dtype=jnp.int16),jax.devices()[0])
    tdelaysall[1::2,:] = (np.array(np.ceil(tdelays/tsamp),dtype=np.int8))
    tdelaysall[0::2,:] = (np.array(np.floor(tdelays/tsamp),dtype=np.int8))
    print("TDELAYSALL:",tdelaysall)
    tdelays_frac = np.zeros((nchans*2,nDM),dtype=float)
    tdelays_frac[0::2,:] = 1 - (tdelays/tsamp - np.floor(tdelays/tsamp))#(np.array(np.ceil(tdelays/tsamp),dtype=np.int8))
    tdelays_frac[1::2,:] = tdelays/tsamp - np.floor(tdelays/tsamp) #(np.array(np.floor(tdelays/tsamp),dtype=np.int8))
    tdelays_frac = tdelays_frac.transpose()
    print("TDELAYSFRAC:",tdelays_frac)
    #tdelays_frac = np.concatenate([tdelays/tsamp - tdelaysall[0::2,:],1 - (tdelays/tsamp - tdelaysall[0::2,:])],axis=0).transpose()
    
    #rearrange shift idxs and expand axes

    #--case 1: appending previous frame
    tDM_max = (4.15)*np.max(DM_trials)*((1/fmin/1e-3)**2 - (1/fmax/1e-3)**2) #ms
    if maxshift is None:
        maxshift = int(np.ceil(tDM_max/tsamp))
    idxs_all = (np.arange(nsamps + maxshift)[:,np.newaxis,np.newaxis]).repeat(nDM,axis=1).repeat(2*nchans,axis=2)
    corr_shifts_all_append = np.array(np.clip(((tdelaysall.transpose()[np.newaxis,:,:].repeat(nsamps + maxshift,axis=0) + idxs_all))%(nsamps+maxshift),a_min=0,a_max=maxshift + nsamps-1)[np.newaxis,np.newaxis,-nsamps:,:,:].repeat(gridsize,axis=0).repeat(gridsize,axis=1),dtype=np.int8)
    tdelays_frac_append = tdelays_frac[np.newaxis,np.newaxis,np.newaxis,:,:].repeat(gridsize,axis=0).repeat(gridsize,axis=1).repeat(nsamps,axis=2)
    if outputwraps:
        wraps_append = ((tdelaysall.transpose()[np.newaxis,:,:].repeat(nsamps + maxshift,axis=0) + idxs_all))[np.newaxis,np.newaxis,-nsamps:,:,:].repeat(gridsize,axis=0).repeat(gridsize,axis=1)<0#>=maxshift+nsamps

    #--case 2: not appending previous frame
    maxshift = 0#int(np.ceil(tDM_max/sl.tsamp))
    idxs_all = (np.arange(nsamps + maxshift)[:,np.newaxis,np.newaxis]).repeat(nDM,axis=1).repeat(2*nchans,axis=2)
    corr_shifts_all_no_append = np.array(np.clip(((tdelaysall.transpose()[np.newaxis,:,:].repeat(nsamps + maxshift,axis=0) + idxs_all))%(nsamps+maxshift),a_min=0,a_max=maxshift + nsamps-1)[np.newaxis,np.newaxis,-nsamps:,:,:].repeat(gridsize,axis=0).repeat(gridsize,axis=1),dtype=np.int8)
    tdelays_frac_no_append = tdelays_frac[np.newaxis,np.newaxis,np.newaxis,:,:].repeat(gridsize,axis=0).repeat(gridsize,axis=1).repeat(nsamps,axis=2)
    if outputwraps:
        wraps_no_append = ((tdelaysall.transpose()[np.newaxis,:,:].repeat(nsamps + maxshift,axis=0) + idxs_all))[np.newaxis,np.newaxis,-nsamps:,:,:].repeat(gridsize,axis=0).repeat(gridsize,axis=1)<0
        return corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append,wraps_append,wraps_no_append
    
    return corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append


#minDM = 171
#maxDM = 4000
DM_trials = np.array(gen_dm(minDM,maxDM,DM_tol,fc*1e-3,nchans,tsamp,chanbw,nsamps))#[0:1]
nDMtrials = len(DM_trials)
corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append = gen_dm_shifts(DM_trials,freq_axis,tsamp,nsamps)

DM_trials_slow = np.array(gen_dm(minDM*5,maxDM*5,DM_tol,fc*1e-3,nchans,tsamp_slow,chanbw,nsamps)) #np.array(gen_dm(minDM*5,maxDM,DM_tol_slow,fc*1e-3,nchans,tsamp_slow,chanbw,nsamps))
nDMtrials_slow = len(DM_trials_slow)
corr_shifts_all_append_slow,tdelays_frac_append_slow,corr_shifts_all_no_append_slow,tdelays_frac_no_append_slow = gen_dm_shifts(DM_trials_slow,freq_axis,tsamp_slow,nsamps)

corr_shifts_all_gpu_0 = copy.deepcopy(corr_shifts_all_append)
tdelays_frac_gpu_0 = copy.deepcopy(tdelays_frac_append)
corr_shifts_all_gpu_1 = copy.deepcopy(corr_shifts_all_append)
tdelays_frac_gpu_1 = copy.deepcopy(tdelays_frac_append)



#make boxcar filters in advance
widthtrials = np.array(2**np.arange(5),dtype=int)
nwidths = len(widthtrials)
def gen_boxcar_filter(widthtrials,truensamps,gridsize=1,nDMtrials=1): #note, you shouldn't need to set gridsize OR DM trials
    nwidths = len(widthtrials)
    boxcar = np.zeros((nwidths,gridsize,gridsize,truensamps,nDMtrials),dtype=np.float16)
    loc = int(truensamps//2)
    for i in range(nwidths):
        wid = widthtrials[i]
        boxcar[i,:,:,:wid,:] = 1 #loc-wid//2:loc+wid-wid//2,:] = 1

    return boxcar
full_boxcar_filter = gen_boxcar_filter(widthtrials,nsamps)
full_boxcar_filter_imgdiff = gen_boxcar_filter(widthtrials,ngulps_per_file)
full_boxcar_filter_gpu_0 = copy.deepcopy(full_boxcar_filter)
full_boxcar_filter_gpu_1 = copy.deepcopy(full_boxcar_filter)

#get the current noise map from file
current_noise = noise_update_all(None,gridsize,gridsize,DM_trials,widthtrials,readonly=True) #noise.get_noise_dict(gridsize,gridsize)

#snr threshold
SNRthresh = 6

#last image frame
tDM_max = (4.15)*np.max(DM_trials)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms
tDM_max_slow = (4.15)*np.max(DM_trials_slow)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms

maxshift = int(np.ceil(tDM_max/tsamp))
maxshift_slow = int(np.ceil(tDM_max_slow/tsamp_slow))
def init_last_frame(gridsize_DEC,gridsize_RA,nsamps,nchans,frame_dir=frame_dir,slow=False):
    noise = np.zeros((gridsize_DEC,gridsize_RA,nsamps,nchans))
    #check if raw noise file exists
    if len(glob.glob(noise_dir + "raw_noise_" + str(gridsize_DEC) + "x" + str(gridsize_RA) + ".npy")) > 0:
        raw_noise = np.load(noise_dir + "raw_noise_" + str(gridsize_DEC) + "x" + str(gridsize_RA) + ".npy")
        for i in range(nchans):
            noise[:,:,:,i] = norm.rvs(loc=0,scale=raw_noise[i],size=(gridsize_DEC,gridsize_RA,nsamps))
    f = open(frame_dir + "last_frame" + str("_slow" if slow else "") + ".npy","wb")
    np.save(f,noise)
    f.close()
    return

"""
#pulse period trials
trial_p_samp = np.array([5, 6, 9, 10, 15],dtype=int)
def gen_psamp_trials(trial_p_samp,nsamp=int(config.ngulps_per_file//config.bin_imgdiff)):
    idxs_full = np.zeros((1,1,1,nsamp,nsamp,len(trial_p_samp)))#np.zeros_like(timeseries,dtype=int)[...,np.newaxis,np.newaxis].repeat(timeseries.shape[-1],-2).repeat(len(trial_p_samp),-1)
    for i in range(len(trial_p_samp)):
        idxs = (np.array([trial_p_samp[i]*np.arange(0,nsamp//trial_p_samp[i],dtype=int)]*trial_p_samp[i],dtype=int) + np.arange(trial_p_samp[i])[:,np.newaxis])
    return idxs_full, idxs_full!=0
P_idxs_full,P_bool_idxs_full = gen_psamp_trials(trial_p_samp)
"""

    

def save_last_frame(image_tesseract,full=False,maxDM=np.max(DM_trials),tsamp=tsamp,frame_dir=frame_dir,slow=False):
    """
    This function writes the given frame to the npy file to store for dedispersion
    on the next timestep
    """

    #if full is set, save the full image; otherwise, only save the samples needed for
    #dedisperion to the maximum value
    if full:
        f = open(frame_dir + "last_frame" + str("_slow" if slow else "") + ".npy","wb")
        np.save(f,image_tesseract)
        f.close()
    else:
        tDM_max = (4.15)*maxDM*((1/fmin/1e-3)**2 - (1/fmax/1e-3)**2) #ms
        maxshift = int(np.ceil(tDM_max/tsamp))
        f = open(frame_dir + "last_frame" + str("_slow" if slow else "") + ".npy","wb")
        np.save(f,image_tesseract[:,:,-maxshift:,:])
        f.close()
def get_last_frame(frame_dir=frame_dir,maxDM=np.max(DM_trials),slow=False):
    f = open(frame_dir + "last_frame" + str("_slow" if slow else "") + ".npy","rb")
    image_tesseract = np.load(f)
    f.close()
    return image_tesseract
try:
    last_frame = get_last_frame()
except Exception as exc:
    printlog(exc,output_file=error_file)
    printlog("initializing last frame to zeros",output_file=error_file)
    last_frame = np.zeros((gridsize,gridsize,nsamps,nchans))
    save_last_frame(last_frame)
last_frame_init_idx = 0

try:
    last_frame_slow = get_last_frame(slow=True)
except Exception as exc:
    printlog(exc,output_file=error_file)
    printlog("initializing slow last frame to zeros",output_file=error_file)
    last_frame_slow = np.zeros((gridsize,gridsize,nsamps,nchans))
    save_last_frame(last_frame_slow,slow=True)
last_frame_slow_init_idx = 0
"""
pre-computed psf values
"""
PSF_dict = scPSF.make_PSF_dict()
default_PSF,default_PSF_params = scPSF.manage_PSF(PSF_dict,gridsize,DEC_axis[int(gridsize//2)],nsamps=nsamps)
default_PSF_gpu_0 = np.array(default_PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(default_PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32)
default_PSF_gpu_1 = np.array(default_PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(default_PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32)

"""
pre-computed cutoff pixels
"""
default_cutoff = get_RA_cutoff(0)


"""Search functions"""

from scipy.signal import convolve2d
from scipy.signal import correlate2d
def matched_filter_space(image_tesseract,PSFimg,kernel_size,usefft=False,device=None,output_file=""):
    """
    Matched filter via convolution w/ DSA-110 core PSF
    """
    """
    if device == None:
        image_tesseract = image_tesseract.from_numpy()
        PSFimg = PSFimg

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        usingGPU = device.type == "cuda"
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    
    #reshape inputs
    nsamps = image_tesseract.shape[2]
    nchans = image_tesseract.shape[3]
    gridsize_RA = image_tesseract.shape[0]
    gridsize_DEC = image_tesseract.shape[1]

    #cut PSF kernel size
    PSF_kernel = PSFimg[gridsize_DEC//2-kernel_size//2:gridsize_DEC//2+kernel_size//2,
                gridsize_RA//2-kernel_size//2:gridsize_RA//2+kernel_size//2,:,:]



    if device != None and device.type=='cuda':
        print("DEVICE: ",device,torch.cuda.is_available(),file=fout)
        #create gpu tensors
        #image_tesseract.to(device)
        #image_tesseract_filtered.to(device)
        
        print("IMG:" + str(image_tesseract) + "," + str(torch.sum(torch.isinf(image_tesseract))) + "," + str(torch.sum(image_tesseract==0)) + "," + str(torch.sum(torch.isnan(image_tesseract))),file=fout)
        print("PSF:" + str(PSF_kernel) + "," + str(torch.sum(torch.isinf(PSF_kernel))) + "," + str(torch.sum(PSF_kernel==0)) + "," + str(torch.sum(torch.isnan(PSF_kernel))),file=fout)
        #fft implementation
        if usefft:
            #take FFT of image and PSF
            image_tesseract_FFT = torch.fft.fft2(image_tesseract.double().to(device),s=(gridsize_DEC,gridsize_RA),dim=(0,1),norm='backward')
            PSF_kernel_FFT = torch.fft.fft2(PSF_kernel.double().to(device),s=(gridsize_DEC,gridsize_RA),dim=(0,1),norm='backward')
            image_tesseract_filtered = torch.real(torch.fft.ifft2(image_tesseract_FFT*PSF_kernel_FFT,s=(gridsize_DEC,gridsize_RA),dim=(0,1),norm='backward')).to("cpu")
            torch.cuda.empty_cache()
            del image_tesseract_FFT
            del PSF_kernel_FFT
            torch.cuda.empty_cache()
            print(image_tesseract_filtered,file=fout)
            



        else:
            #reshape
            image_tesseract_reshaped = (image_tesseract.transpose(0,2).transpose(1,3)).double()
            PSFimg_reshaped = (((PSF_kernel[:,:,0,:].transpose(0,2).transpose(1,2)).unsqueeze(0)).expand(nchans,-1,-1,-1)).double()#to(image_tesseract_reshaped.dtype)
        
            #convolve
            image_tesseract_filtered = tf.conv2d(image_tesseract_reshaped.to(device),PSFimg_reshaped.to(device),padding='same').transpose(1,3).transpose(0,2).to("cpu")
            torch.cuda.empty_cache()
            del image_tesseract_reshaped
            del PSFimg_reshaped
            torch.cuda.empty_cache()
            print(image_tesseract_filtered,file=fout)


    else:
        image_tesseract_filtered = np.zeros(image_tesseract.shape)
        for i in range(nsamps):
            for j in range(nchans):
                if usefft:
                    image_tesseract_filtered[:,:,i,j] =  np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(image_tesseract[:,:,i,j])*np.fft.fft2(PSFimg[:,:,i,j]))))#np.abs(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(np.fft.fft(image_tesseract[:,:,i,j]))*np.fft.fftshift(np.fft.fft(PSFimg[:,:,i,j])))))
                else:
                    image_tesseract_filtered[:,:,i,j] = convolve2d(image_tesseract[:,:,i,j],PSFimg[:,:,i,j],mode='same') #assume the PSF is already centered


    if output_file != "":
        fout.close()

    return image_tesseract_filtered
    
    
    #np.nansum(np.nansum((img/np.array(noises)),3)*np.nanmean(PSFimg,3)/(np.nansum(1/np.array(noises))),axis=(0,1))


def snr_vs_RA_DEC_new(image_tesseract_filtered_dm,wid,DM,mode='4d',noiseth=0.9,samenoise=False,plot=False,device=None,output_file="",scrunch=False,exportmaps=False,usefft=False):
    """
    alternate implementation of SNR w/ 2d convolution to do PSF matched filtering. input is 3d array with axes gridsize x gridsize x nsamps
    """

    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    nsamps = image_tesseract_filtered_dm.shape[2]
    #ndms = image_tesseract.shape[3]
    gridsize_RA = image_tesseract_filtered_dm.shape[1]
    gridsize_DEC = image_tesseract_filtered_dm.shape[0]
    loc = nsamps//2
    
    if device != None and device.type=='cuda':
        print(torch.cuda.is_available(),file=fout)

        #make tensors for GPU
        boxcar = torch.zeros(image_tesseract_filtered_dm.shape[2])
        image_tesseract_binned = torch.zeros((gridsize_DEC,gridsize_RA))
        noisemap = torch.zeros((gridsize_DEC,gridsize_RA))
        csig_all_flat = torch.zeros((gridsize_RA*gridsize_DEC,1,nsamps))#image_tesseract_filtered_dm.shape)

        #image_tesseract_filtered_dm.to(device)
        #boxcar.to(device)
        #image_tesseract_binned.to(device)
        #noisemap.to(device)
        

        #make a boxcar filter for time
        boxcar[loc-wid//2-2:loc+wid-wid//2-2] = 1

        if plot:
            plt.figure(figsize=(40,12))
            plt.subplot(1,4,2)
            plt.plot(boxcar)
        
        #convolve for each timeseries; assume already normalized

        #reshape input to be [batch_size, channels, sequence_length] = [gridsize*gridsize,1,nsamps]
        image_tesseract_reshaped = image_tesseract_filtered_dm[:,:,np.newaxis,:].reshape((gridsize_RA*gridsize_DEC,1,nsamps))#image_tesseract_filtered_dm.reshape((gridsize_RA*gridsize_DEC,1,nsamps))

        #reshape boxcar to be [channels,channels/groups,sequence_length] = [1,1,nsamps]
        boxcar_reshaped = boxcar[np.newaxis,np.newaxis,:]#.reshape((1,1,nsamps))
        boxcar_reshaped.to(device)

        #convolve and get peak value
        maxbatchsize = gridsize_DEC*gridsize_RA 
        nbatches = 1#gridsize_RA
        
        for i in range(nbatches):
            #move subset of pixels to gpu
            subimg = image_tesseract_reshaped[i*maxbatchsize:(i+1)*maxbatchsize,:,:]
            subimg.to(device)
            
            #convolve
            if scrunch:
                if nsamps%wid != 0: subimg = subimg[:,:,:nsamps - nsamps%wid]
                csig_all_flat[i*maxbatchsize:(i+1)*maxbatchsize,:,:(nsamps - nsamps%wid)//wid] = subimg.reshape((maxbatchsize,1,(nsamps - nsamps%wid)//wid,wid)).mean(3)
            else:
                csig_all_flat[i*maxbatchsize:(i+1)*maxbatchsize,:,:] = tf.conv1d(torch.nan_to_num(subimg.double()).double(),boxcar_reshaped.double(),padding='same')
            
        subimg.to("cpu")
        boxcar_reshaped.to("cpu")
        del subimg
        del boxcar_reshaped
        torch.cuda.empty_cache()
        csig_all = csig_all_flat.reshape((gridsize_RA,gridsize_DEC,nsamps))

        #peakidx = torch.argmax(csig_all,dim=2)

        #noise estimate
        csig_all.to(device)
        
        #mask all nans and values less than noise threshold
        print("CSIG:" + str(csig_all.numpy()),file=fout)
        mask1 = torch.logical_not(torch.logical_or(torch.isinf(csig_all),torch.isnan(csig_all)))
        csig_all = torch.nan_to_num(csig_all)
        csig_all[torch.logical_or(torch.isinf(csig_all),torch.isnan(csig_all))] = 0


        print("MASK1:" + str(mask1.numpy()) + "," + str(mask1.sum()),file=fout)
        print("QUANTILES:" + str(torch.nanquantile(csig_all,noiseth)),file=fout)#,dim=2,keepdim=True).numpy()),file=fout)
        mask = (csig_all) < torch.nanquantile(csig_all,noiseth)#,dim=2,keepdim=True) #(noiseth*(torch.max(csig_all,dim=2,keepdim=True).values))
        
        
        print("MASK:" + str(mask.numpy()) + "," + str(mask.sum()),file=fout)
        csig_all_masked = csig_all*mask
        numvalids = (mask*mask1).sum(2)
        print("NONMASKED VALS:" + str(csig_all_masked.numpy()),file=fout)

        #take std deviation and correct for nan and non-noise data
        noisemap = torch.std(csig_all_masked,dim=2)*torch.sqrt(nsamps/numvalids)
        noise = float(torch.mean(noisemap[~torch.logical_or(torch.isinf(noisemap),torch.isnan(noisemap))]).numpy())
        tmp,noise = noise_update(noise,gridsize_RA,gridsize_DEC,DM,wid)
        print("NOISE:"+str(noise) + ", " + str(noisemap.numpy()),file=fout)
        
        #take off-pulse median
        meanmap = torch.median(csig_all_masked,dim=2).values
        print("MEDIAN:"+str(meanmap.numpy()),file=fout)
        
        #get snr
        image_tesseract_binned = (csig_all.max(2).values - meanmap)/noise
       

        csig_all.to("cpu")
        csig_all_masked.to("cpu")
        noisemap.to("cpu")
        if exportmaps:
            f = open(noise_dir + "noisemap_" + str(gridsize_RA) + "x" + str(gridsize_DEC) + "_DM" + str(DM)+ "_W" + str(wid) + ".npy","wb")
        meanmap.to("cpu")
        image_tesseract_binned.to("cpu")
        numvalids.to("cpu")
        del csig_all
        del csig_all_masked
        del noisemap
        del meanmap
        del numvalids
        torch.cuda.empty_cache()
        print("FINAL ARRAY:" + str(image_tesseract_binned.numpy()),file=fout)        
        print(np.max(image_tesseract_binned.numpy()),file=fout)
    else:
        #make a boxcar filter for time
        boxcar = np.zeros(image_tesseract_filtered_dm.shape[2])
        boxcar[loc-wid//2-2:loc+wid-wid//2-2] = 1

        if plot:
            plt.figure(figsize=(40,12))
            plt.subplot(1,4,2)
            plt.plot(boxcar)
        #convolve for each timeseries; assume already normalized
        image_tesseract_binned = np.zeros((gridsize_DEC,gridsize_RA))
        noisemap=np.zeros((gridsize_DEC,gridsize_RA))
        csig_all = np.zeros(image_tesseract_filtered_dm.shape)

        for i in range(gridsize_DEC):
            for j in range(gridsize_RA):
                timeseries = image_tesseract_filtered_dm[i,j,:]
                csig_all[i,j,:] = np.convolve(np.nan_to_num(timeseries,nan=0),boxcar,'same')

        csig_all[np.isinf(csig_all)] = np.nan
        mask = csig_all < np.nanpercentile(csig_all,noiseth*100)
        print("MASK:" + str(mask) + "," + str(mask.sum()),file=fout)
        
        csig_all_masked = copy.deepcopy(csig_all)
        csig_all_masked[~mask] = np.nan
        numvalids = np.nansum(mask,axis=2)
        print("NONMASKED VALS:" + str(csig_all[mask]),file=fout)

        #take std deviation and correct for nan and non-noise data
        noisemap = np.nanstd(csig_all_masked,axis=2)
        noisemap[np.isinf(noisemap)] = np.nan
        noise = float(np.nanmean(noisemap))
        tmp,noise = noise_update(noise,gridsize_RA,gridsize_DEC,DM,wid)
        print("NOISE:"+str(noise) + ", " + str(noisemap),file=fout)

        #take off-pulse median
        meanmap = np.nanmedian(csig_all_masked,axis=2)
        print("MEDIAN:"+str(meanmap),file=fout)

        #get snr
        image_tesseract_binned = (csig_all.max(axis=2) - meanmap)/noise

        if exportmaps:
            f = open(noise_dir + "noisemap_" + str(gridsize_RA) + "x" + str(gridsize_DEC) + "_DM" + str(DM)+ "_W" + str(wid) + ".npy","wb")
            np.save(f,noisemap)
            f.close()
    #TMP: save noise statistics
    #np.save("noisestats.npy",noisemap)

    if plot:
        plt.subplot(1,4,4)
        plt.hist(noisemap.flatten())
        plt.axvline(noiseth/np.sqrt(wid))
        #plt.hist(np.std(image_tesseract_filtered_dm[:,:,:25//2],axis=2).flatten(),np.linspace(0,10,100))
        
        plt.subplot(1,4,3)
        plt.axhline(noiseth/np.sqrt(wid))
        #plt.axvline(loc + wid,color='red')
        #plt.axhline(1000,color='red')
        plt.show()

    if output_file != "":
        fout.close()
    
    return image_tesseract_binned


def snr_vs_RA_DEC_allDMW(image_tesseract_filtered_dm,DM_trials=DM_trials,widthtrials=widthtrials,mode='4d',noiseth=0.9,samenoise=False,plot=False,device=None,output_file="",scrunch=False,exportmaps=False,usefft=False,batches=1,usejax=False,maxProcesses=5):
    """
    boxcar convolution, input is 4d array with axes gridsize x gridsize x nsamps  x nDMtrials
    """

    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    nsamps = image_tesseract_filtered_dm.shape[2]
    ndms = image_tesseract_filtered_dm.shape[3]
    nwidths = len(widthtrials)
    gridsize_RA = image_tesseract_filtered_dm.shape[1]
    gridsize_DEC = image_tesseract_filtered_dm.shape[0]
    subgridsize_RA = int(gridsize_RA//batches)
    subgridsize_DEC = int(gridsize_DEC//batches)
    loc = nsamps//2


    if device != None and device.type=='cuda':
        #make tensors for GPU
        boxcar = torch.zeros(image_tesseract_filtered_dm.shape).unsqueeze(0).expand(nwidths,-1,-1,-1,-1)
        for i in range(nwidths):
            wid = widthtrials[i]
            boxcar[i,:,:,loc-wid//2-2:loc+wid-wid//2-2,:] = 1
        #print("BOXCAR SHAPE " + str(boxcar.shape),file=fout)
        if usefft:
            if usejax:
                image_tesseract_binned = torch.zeros((gridsize_DEC,gridsize_RA,nwidths,ndms))
                #image_tesseract_binned = jax.device_put(jnp.zeros((gridsize_DEC,gridsize_RA,nwidths,ndms)),jax.devices("cpu")[0])
                total_noise = torch.zeros((nwidths,ndms))
                prev_noise,prev_noise_N = noise_update_all(None,gridsize_RA,gridsize_DEC,DM_trials,widthtrials,readonly=True)
                
                
                executor = ThreadPoolExecutor(maxProcesses)#ProcessPoolExecutor(args.maxProcesses)
                task_list = []
                for i in range(batches):
                    for j in range(0,batches,2):
                        #take fourier transform of boxcar and image

                        task_list.append(executor.submit(jax_funcs.inner_snr_fft_jit_0,np.array(image_tesseract_filtered_dm[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].numpy(),dtype=np.float64),np.array(boxcar[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].numpy(),dtype=np.float64),np.array(prev_noise,dtype=np.float64),prev_noise_N,noiseth,i,j))
                        #outtup = jax_funcs.inner_snr_fft_jit_0(np.array(image_tesseract_filtered_dm[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].numpy(),dtype=np.float64),np.array(boxcar[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].numpy(),dtype=np.float64),np.array(prev_noise,dtype=np.float64),prev_noise_N,noiseth)
                        #image_tesseract_binned[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = torch.from_numpy(np.array(outtup[0]))
                        #total_noise += torch.from_numpy(np.array(outtup[1]))/(batches**2)

                        if j+1 < batches:
                            task_list.append(executor.submit(jax_funcs.inner_snr_fft_jit_1,np.array(image_tesseract_filtered_dm[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j+1)*subgridsize_RA:(j+1+1)*subgridsize_RA,:,:].numpy(),dtype=np.float64),np.array(boxcar[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j+1)*subgridsize_RA:(j+1+1)*subgridsize_RA,:,:].numpy(),dtype=np.float64),np.array(prev_noise,dtype=np.float64),prev_noise_N,noiseth,i,j+1))
                            #outtup = jax_funcs.inner_snr_fft_jit_1(np.array(image_tesseract_filtered_dm[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].numpy(),dtype=np.float64),np.array(boxcar[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].numpy(),dtype=np.float64()),np.array(prev_noise,dtype=np.float64),prev_noise_N,noiseth)
                            #image_tesseract_binned[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j+1)*subgridsize_RA:(j+1+1)*subgridsize_RA,:,:] = torch.from_numpy(np.array(outtup[0]))
                            #total_noise += torch.from_numpy(np.array(outtup[1]))/(batches**2)


                        #outtup = task_list[0].result()
                        #image_tesseract_binned[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = torch.from_numpy(np.array(outtup[0]))
                        #total_noise += torch.from_numpy(np.array(outtup[1]))/(batches**2)
                            
                        #if j+1 < batches:
                        #    outtup = task_list[1].result()
                        #    image_tesseract_binned[i*subgridsize_DEC:(i+1)*subgridsize_DEC,(j+1)*subgridsize_RA:(j+1+1)*subgridsize_RA,:,:] = torch.from_numpy(np.array(outtup[0]))
                        #    total_noise += torch.from_numpy(np.array(outtup[1]))/(batches**2)   
                            
                            
                        #print("NOISE:" + str(torch.from_numpy(np.array(outtup[1]))/(batches**2)),file=fout)
                
                
                for t in task_list:
                    outtup = t.result()
                    i,j = outtup[2],outtup[3]
                    image_tesseract_binned[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = torch.from_numpy(np.array(outtup[0]))
                    total_noise += torch.from_numpy(np.array(outtup[1])/(batches**2))
                executor.shutdown()
                
                image_tesseract_binned = torch.from_numpy(np.array(image_tesseract_binned))
                #noise_update_all(total_noise.numpy(),gridsize_RA,gridsize_DEC,DM_trials,widthtrials,writeonly=True) 
                #print("BINNED IMG SHAPE:" + str(image_tesseract_binned.shape),file=fout)
                #print("BINNED IMG:" + str(image_tesseract_binned),file=fout)
                
            else:
                image_tesseract_binned = torch.zeros((nwidths,gridsize_DEC,gridsize_RA,ndms,nsamps)).to(device)
                for i in range(batches):
                    for j in range(batches):

                        #take fourier transform of boxcar and image
                        image_tesseract_binned[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = torch.real(torch.fft.ifftshift(
                                                                                                                                        torch.fft.ifft(
                                                                                                                                            torch.fft.fft(
                                                                                                                                                image_tesseract_filtered_dm[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].to(device),
                                                                                                                                            n=nsamps,dim=2,norm='backward')*torch.fft.fft(
                                                                                                                                                boxcar[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].to(device),n=nsamps,dim=3,norm='backward'),
                                                                                                                                            n=nsamps,dim=3,norm='backward'),dim=3)).transpose(3,4) ##output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps
        
                print("BINNED IMG SHAPE:" + str(image_tesseract_binned.shape),file=fout)
                print("BINNED IMG:" + str(image_tesseract_binned),file=fout)
        else:
            image_tesseract_binned = torch.zeros((nwidths,gridsize_DEC,gridsize_RA,ndms,nsamps)).to(device)
            for i in range(batches):
                for j in range(batches):
                    #convolve for each timeseries; assume already normalized; JAX takes too long, always use pytorch
                    #if usejax:
                    #    image_tesseract_binned[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = torch.from_numpy(np.array(jax_funcs.inner_snr_conv_jit(image_tesseract_filtered_dm[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].numpy(),boxcar[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].numpy()))).to(device)
                    #else:
                    image_tesseract_binned[:,i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:] = tf.conv1d(image_tesseract_filtered_dm[i*subgridsize_DEC:(i+1)*subgridsize_DEC,j*subgridsize_RA:(j+1)*subgridsize_RA,:,:].transpose(2,3).reshape((subgridsize_RA*subgridsize_DEC*ndms,1,nsamps)).to(device),
                                                        boxcar[:,0,0,:,0:1].transpose(1,2).to(device).to(image_tesseract_filtered_dm.dtype),padding='same').reshape(subgridsize_DEC,subgridsize_RA,ndms,nwidths,nsamps).transpose(0,3).transpose(2,3).transpose(1,2) #output of shape nwidths x gridsize_DEC x gridsize_RA x ndms x nsamps
        
        
        if (not usefft) or (usefft and not usejax): 
            print("BINNED IMG SHAPE:" + str(image_tesseract_binned.shape),file=fout)

            torch.cuda.empty_cache()
            #np.save("tmp.npy",image_tesseract_binned.to("cpu").numpy())

            #np.save("tmp_DM.npy",image_tesseract_filtered_dm.to("cpu").numpy())
            #mask all nans and values less than noise threshold
            #print("CSIG:" + str(csig_all.numpy()),file=fout)
            mask1 = torch.logical_not(torch.logical_or(torch.isinf(image_tesseract_binned),torch.isnan(image_tesseract_binned)))
            image_tesseract_binned[:] = torch.nan_to_num(image_tesseract_binned[:])
            image_tesseract_binned[torch.logical_or(torch.isinf(image_tesseract_binned),torch.isnan(image_tesseract_binned))] = 0


            print("MASK1:" + str(mask1) + "," + str(mask1.sum()),file=fout)
            #print("QUANTILES:" + str(torch.nanquantile(image_tesseract_binned,noiseth)),file=fout)#,dim=2,keepdim=True).numpy()),file=fout)
            #mask2 = ((image_tesseract_binned) < torch.nanquantile((image_tesseract_binned.max(dim=4,keepdim=True).values).reshape((nwidths,gridsize_RA*gridsize_DEC,ndms,1)),noiseth,dim=1,keepdim=True).unsqueeze(1))#.reshape(mask1.shape)

            mask2 = ((image_tesseract_binned) < torch.nanquantile(torch.flatten(image_tesseract_binned.max(dim=4,keepdim=True).values,1,2),noiseth,dim=1,keepdim=True).unsqueeze(1))


        
            print("MASK2:" + str(mask2) + "," + str(mask2.sum()) + "," + str(mask2.shape),file=fout)
        
            #torch.nanquantile(image_tesseract_binned,noiseth)#,dim=2,keepdim=True) #(noiseth*(torch.max(csig_all,dim=2,keepdim=True).values))
            mask = (mask2*mask1).sum(4,keepdims=True) > 1 
            masknan = (mask2*mask1).sum(4,keepdims=True) > 1
            masknan[masknan==0] = torch.nan

            print("MASK:" + str(mask) + "," + str(mask.sum()),file=fout)
            #image_tesseract_binned_masked = image_tesseract_binned*mask
            numvalids = (mask2*mask1).sum(4)
            print("NONMASKED VALS:" + str(image_tesseract_binned*mask1*mask2) + "," + str(numvalids.shape),file=fout)
            print("STD:" + str(torch.std(
                                                image_tesseract_binned*mask1*mask2,dim=4
                                                )*torch.sqrt(nsamps/numvalids)),file=fout)
            print("MEDIAN NOISE:" + str(torch.nanmedian(
                                        torch.nanmedian(
                                            (torch.std(
                                                image_tesseract_binned*mask1*mask2,dim=4
                                                )*torch.sqrt(nsamps/numvalids)
                                            ),dim=1
                                        ).values,dim=1
                                    )),file=fout)
            print("ALT MEDIAN NOISE:" + str(torch.nanmedian(
                                        torch.nanmedian(
                                            ((torch.std(
                                                image_tesseract_binned*mask1*mask2,dim=4
                                                )*torch.sqrt(nsamps/numvalids))*masknan[:,:,:,:,0]
                                            ),dim=1
                                        ).values,dim=1
                                        )),file=fout)
        
            #take std deviation and correct for nan and non-noise data
            print(image_tesseract_binned.shape,mask.shape,numvalids.shape)
            noise = torch.from_numpy(noise_update_all(
                                    torch.nanmedian(
                                        torch.nanmedian(
                                            (torch.std(
                                                image_tesseract_binned*mask1*mask2,dim=4
                                                )*torch.sqrt(nsamps/numvalids)
                                            ),dim=1
                                        ).values,dim=1
                                    ).values.to("cpu").numpy(),
                                gridsize_RA,gridsize_DEC,DM_trials,widthtrials)).to(device)
            print("NOISE:"+str(noise),file=fout)

            #get snr
        
            print(str(image_tesseract_binned.shape) + " " + str(mask.shape) + " " + str(numvalids.shape),file=fout)
            print("HEEEERE:" + str(image_tesseract_binned.shape) + "," + str(noise.shape),file=fout)
            image_tesseract_binned = ((image_tesseract_binned.max(4).values - torch.nanmedian(image_tesseract_binned*mask1*mask2,dim=4).values)/noise.unsqueeze(1).unsqueeze(1)).cpu()#to("cpu")
            print("IMG BINNED FINAL:" + str(image_tesseract_binned),file=fout)
            del mask
            del mask1
            del numvalids
            del noise
            torch.cuda.empty_cache()

            image_tesseract_binned = image_tesseract_binned.transpose(0,2).transpose(0,1)

    else:
        image_tesseract_binned = np.zeros((gridsize_DEC,gridsize_RA,nwidths,ndms))
        for i in range(nwidths):
            wid = widthtrials[i]
            for j in range(ndms):
                DM = DM_trials[j]
                image_tesseract_binned[:,:,i,j] = snr_vs_RA_DEC_new(image_tesseract_filtered_dm[:,:,:,j],wid,DM,mode=mode,noiseth=noiseth,samenoise=samenoise,plot=plot,device=device,output_file=output_file,scrunch=scrunch,exportmaps=exportmaps,usefft=usefft)

                


    return image_tesseract_binned




# Brute force dedispersion
def dedisperse(image_tesseract_point,DM,tsamp=tsamp,freq_axis=freq_axis,device=None,output_file="",append_last_frame=True):
    """
    [Deprecated]
    This function dedisperses a dynamic spectrum pixel grid of shape gridsize x gridsize x nsamps x nchans by brute force without accounting for edge effects
    """

    """#find available cuda GPU to use
    if device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        usingGPU = device.type == "cuda"
    """

    


    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout


    #get data from previous timeframe
    if append_last_frame:
        truensamps = image_tesseract_point.shape[2]
        if device != None and device.type == 'cuda':
            image_tesseract_point= torch.cat([torch.from_numpy(get_last_frame()),image_tesseract_point],dim=2)
        else:
            image_tesseract_point = np.concatenate([get_last_frame(),image_tesseract_point],axis=2)
        print("Appending data from previous timeframe, new shape: " + str(image_tesseract_point.shape),file=fout)

    #get delay axis
    neg_flag = DM < 0
    DM = np.abs(DM)

    if device != None and device.type == 'cuda':
        #make cuda tensors
        print(torch.cuda.is_available())
        print(image_tesseract_point.shape)
        freq_axis = torch.from_numpy(freq_axis)
        dedisp_timeseries_all = torch.zeros(image_tesseract_point.shape[:-1])
        dedisp_img = torch.zeros(image_tesseract_point.shape)
        #image_tesseract_point.to(device)
        #freq_axis.to(device)
        #dedisp_timeseries_all.to(device)
        #dedisp_img.to(device)


        #Delays
        tdelays = DM*4.15*(((torch.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
        tdelays_idx_hi = torch.ceil(tdelays/tsamp).int()
        tdelays_idx_low = torch.floor(tdelays/tsamp).int()
        tdelays_frac = tdelays/tsamp - tdelays_idx_low
        print("Trial DM: " + str(DM) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout,end="")
        nchans = len(freq_axis)
        nsamps = image_tesseract_point.shape[-2]

        #rearrange shift idxs
        idxs_all = torch.arange(nsamps).to(device).unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,16)
        shifts_all_hi = -tdelays_idx_hi.to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],nsamps,-1)
        shifts_all_low = -tdelays_idx_low.to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],nsamps,-1)
        corr_shifts_all_hi = (idxs_all - shifts_all_hi)%nsamps
        corr_shifts_all_low = (idxs_all - shifts_all_low)%nsamps
        mask = ~torch.logical_or(corr_shifts_all_hi < 0, corr_shifts_all_low < 0)

        #shift, sum but mask the ones with negative indices
        dedisp_img = torch.gather(image_tesseract_point.to(device),dim=2,index=corr_shifts_all_hi)*(tdelays_frac.to(device)) + torch.gather(image_tesseract_point.to(device),dim=2,index=corr_shifts_all_low)*(1-tdelays_frac.to(device))
        dedisp_timeseries_all = (dedisp_img*mask).sum(3)

        del idxs_all
        del shifts_all_hi
        del shifts_all_low
        del corr_shifts_all_hi
        del corr_shifts_all_low
        del mask
      

        dedisp_img = dedisp_img.to("cpu")
        dedisp_timeseries_all = dedisp_timeseries_all.to("cpu")
        torch.cuda.empty_cache()
    else:

        #Delays
        tdelays = DM*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
        tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
        tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
        tdelays_frac = tdelays/tsamp - tdelays_idx_low
        print("Trial DM: " + str(DM) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout,end="")
        nchans = len(freq_axis)
        nsamps = image_tesseract_point.shape[-2]
        
        #shift each channel
        dedisp_timeseries_all = np.zeros(image_tesseract_point.shape[:-1])
        dedisp_img = np.zeros(image_tesseract_point.shape)
        #shift each channel
        for k in range(nchans):
            #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac);
            if neg_flag:
                padshape = tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-1) + [(tdelays_idx_low[k],0)])
                arrlow =  np.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",constant_values=0)[:,:,:nsamps]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
            else:
                padshape = tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-1) + [(0,tdelays_idx_low[k])])
                arrlow =  np.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",constant_values=0)[:,:,tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
            print(padshape,file=fout)

            if neg_flag:
                padshape = tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-1) + [(tdelays_idx_hi[k],0)])
                arrhi =  np.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",constant_values=0)[:,:,:nsamps]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
            else:
                padshape = tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-1) + [(0,tdelays_idx_hi[k])])
                arrhi =  np.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",constant_values=0)[:,:,tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

            print(padshape,file=fout)

            dedisp_timeseries_all += arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])
            dedisp_img[:,:,:,k] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])
    
    #save frame
    if append_last_frame:
        save_last_frame(image_tesseract_point)
        print("Writing to last_frame.npy",file=fout)


    print("Done!",file=fout)
    if output_file != "":
        fout.close()
    if append_last_frame:
        return dedisp_timeseries_all[:,:,:truensamps], dedisp_img[:,:,:truensamps,:]
    return dedisp_timeseries_all,dedisp_img


def dedisperse_allDM(image_tesseract_point,DM_trials,tsamp=tsamp,freq_axis=freq_axis,device=None,output_file="",append_frame=True,_idx=0,multithreading=False,maxProcesses=1,DMbatches=1,usejax=True,keepfreqaxis=False):
    """
    This function dedisperses a dynamic spectrum pixel grid of shape gridsize x gridsize x nsamps x nchans by brute force without accounting for edge effects
    """

    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #if device != None and device.type == 'cuda':
    #    if device.index == 0: device = torch.device(1)
    #    else: device = torch.device(0)

    #get data from previous timeframe
    if append_frame:
        truensamps = image_tesseract_point.shape[2]
        if device != None and device.type == 'cuda' and not usejax:
            print("YOLOLOL",get_last_frame().shape,image_tesseract_point.shape)
            image_tesseract_point= torch.cat([torch.from_numpy(get_last_frame()),image_tesseract_point],dim=2)
        else:
            image_tesseract_point = np.concatenate([get_last_frame(),image_tesseract_point],axis=2)
        print("Appending data from previous timeframe, new shape: " + str(image_tesseract_point.shape),file=fout)

    #get delay axis
    #neg_flag = DM < 0
    #DM = np.abs(DM)
    if device != None and device.type == 'cuda' and usejax:
        dedisp_timeseries_all = np.zeros((image_tesseract_point.shape[0],image_tesseract_point.shape[1],image_tesseract_point.shape[2],len(DM_trials)))
        dedisp_img = np.zeros((image_tesseract_point.shape[0],image_tesseract_point.shape[1],image_tesseract_point.shape[2],image_tesseract_point.shape[3],len(DM_trials)))
        gridsize = image_tesseract_point.shape[0]
        subgridsize = gridsize//DMbatches

        if not keepfreqaxis:
            executor = ThreadPoolExecutor(5)
            task_list = []
        for j in range(DMbatches):
            for i in range(DMbatches):
                if keepfreqaxis:
                    dedisp_img[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:,:] = jax_funcs.inner_dedisperse_keepfreqaxis_jit(image_tesseract_point[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:],DM_trials_in=DM_trials,tsamp=tsamp,freq_axis_in=freq_axis)#,fout=fout)
                else:
                    if i%2 == 0:
                        print("DEVICE " + str(int(i%2)),file=fout)
                        #dedisp_timeseries_all[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:] = jax_funcs.inner_dedisperse_jit_0(image_tesseract_point[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:],DM_trials_in=DM_trials,tsamp=tsamp,freq_axis_in=freq_axis)#,fout=fout)
                        task_list.append(executor.submit(jax_funcs.inner_dedisperse_jit_0,np.array(image_tesseract_point[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:],dtype=np.float32),DM_trials,tsamp,freq_axis,i,j))
                    else:
                        print("DEVICE " + str(int((i+1)%2)),file=fout)
                        #dedisp_timeseries_all[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:] = jax_funcs.inner_dedisperse_jit_1(image_tesseract_point[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:],DM_trials_in=DM_trials,tsamp=tsamp,freq_axis_in=freq_axis)#,fout=fout)
                        task_list.append(executor.submit(jax_funcs.inner_dedisperse_jit_0,np.array(image_tesseract_point[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:],dtype=np.float32),DM_trials,tsamp,freq_axis,i,j))

        if not keepfreqaxis:
            for t in task_list:
                dat,i,j = t.result()
                dedisp_timeseries_all[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:] = dat
            executor.shutdown()
    elif device != None and device.type == 'cuda':
        
        
   
        #make cuda tensors
        print(torch.cuda.is_available())
        print(image_tesseract_point.shape)
        freq_axis = torch.from_numpy(freq_axis).to(device)
        #dedisp_timeseries_all = torch.zeros(image_tesseract_point.shape[:-1])
        #dedisp_img = torch.zeros(image_tesseract_point.shape).unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))
        DM_trials = torch.from_numpy(DM_trials).to(device)
        #add axes for DM trials
        #dedisp_timeseries_all = dedisp_timeseries_all.unsqueeze(3).expand(-1,-1,-1,len(DM_trials))
        #image_tesseract_point_DM = image_tesseract_point.unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials)).to(device)
                


        #Delays
        gridsize = image_tesseract_point.shape[0]
        print("SIZES:" + str(gridsize) + "," + str(gridsize//DMbatches) + "," + str(DMbatches),file=fout)
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
        #del tdelays
        #del freq_axis


        #rearrange shift idxs
        """
        idxs_all = (torch.arange(nsamps).unsqueeze(1).unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,16,len(DM_trials))).to(device)
        corr_shifts_all_hi = -tdelays_idx_hi.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],nsamps,-1,-1)
        corr_shifts_all_low = -tdelays_idx_low.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],nsamps,-1,-1)
        """
        idxs_all = (torch.arange(nsamps).unsqueeze(1).unsqueeze(1)).expand(-1,nchans,len(DM_trials)).to(device)
        corr_shifts_all_hi = torch.clip(((-tdelays_idx_hi.unsqueeze(0).expand(nsamps,-1,-1) + idxs_all))%nsamps,min=0,max=nsamps-1)#(idxs_all - shifts_all_hi)%nsamps
        corr_shifts_all_low = torch.clip(((-tdelays_idx_low.unsqueeze(0).expand(nsamps,-1,-1) + idxs_all))%nsamps,min=0,max=nsamps-1)#(idxs_all - shifts_all_low)%nsamps
        #corr_shifts_all_hi = corr_shifts_all_hi.long().unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,-1,-1)
        #corr_shifts_all_low = corr_shifts_all_low.long().unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,-1,-1)
        #mask = ~torch.logical_or(corr_shifts_all_hi < 0, corr_shifts_all_low < 0)

        #shift, sum but mask the ones with negative indices
        #tdelays_frac = tdelays_frac.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],nsamps,-1,-1)

        print("TDEL_HI:" + str(corr_shifts_all_hi),file=fout)
        print("TDEL_LOW:" + str(corr_shifts_all_low),file=fout)
        
        #torch.cuda.empty_cache()
        #del idxs_all


        """
        dedisp_img_hi = (torch.gather(image_tesseract_point_DM.half().to(device),dim=2,index=torch.clip(corr_shifts_all_hi.to(device),min=0,max=nsamps-1))).to("cpu")
        torch.cuda.empty_cache()
        dedisp_img_hi = (dedisp_img_hi.to(device)*tdelays_frac.half().to(device)).to("cpu")
        torch.cuda.empty_cache()
        dedisp_img_low = (torch.gather(image_tesseract_point_DM.half().to(device),dim=2,index=torch.clip(corr_shifts_all_low.to(device),min=0,max=nsamps-1))).to("cpu")
        torch.cuda.empty_cache()
        dedisp_img_low = (dedisp_img_low.to(device)*(1 - tdelays_frac.half().to(device))).to("cpu")
        torch.cuda.empty_cache()
        dedisp_img += (dedisp_img_low.to(device) + dedisp_img_hi.to(device)).to("cpu")
        torch.cuda.empty_cache()
        """
        
        if multithreading and not torch.cuda.is_available():
            tdelays_frac = tdelays_frac.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],nsamps,-1,-1)
            #just multithread the gather stages
            corr_shifts_all_hi = corr_shifts_all_hi.long().unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,-1,-1)
            corr_shifts_all_low = corr_shifts_all_low.long().unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,-1,-1)
            image_tesseract_point_DM = image_tesseract_point.unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))
            print("IMG:"+str(image_tesseract_point_DM),file=fout)
            dedisp_img_hi = torch.zeros(image_tesseract_point.shape).unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))
            dedisp_img_low = torch.zeros(image_tesseract_point.shape).unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))


            executor = ThreadPoolExecutor(maxProcesses)
            task_list_hi = []
            task_list_low = []
            subgridsize = gridsize//maxProcesses
            k = 0
            ks = torch.zeros((maxProcesses,maxProcesses))
            for j in range(maxProcesses):
                for i in range(maxProcesses):
                    task_list_hi.append(executor.submit(torch.gather,
                        image_tesseract_point_DM.half()[j*subgridsize:(j+1)*subgridsize,
                                                    i*subgridsize:(i+1)*subgridsize,:,:,:].to("cpu"),
                                                    2,corr_shifts_all_hi.to("cpu")))
                    task_list_low.append(executor.submit(torch.gather,
                        image_tesseract_point_DM.half()[j*subgridsize:(j+1)*subgridsize,
                                                    i*subgridsize:(i+1)*subgridsize,:,:,:].to("cpu"),
                                                    2,corr_shifts_all_low.to("cpu")))
                    ks[j,i] = k
            for j in range(maxProcesses):
                for i in range(maxProcesses):
                    dedisp_img_hi[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:,:] = task_list_hi[ks[j,i]].result()[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:,:]
                    dedisp_img_low[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:,:] = task_list_low[ks[j,i]].result()[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:,:]
            executor.shutdown()
            #dedisp_img_hi = (dedisp_img_hi.to(device)*tdelays_frac.half().to(device)).to("cpu")
            #dedisp_img_low = (dedisp_img_low.to(device)*(1 - tdelays_frac.half().to(device))).to("cpu")
            if keepfreqaxis:
                dedisp_img = ((dedisp_img_hi.to(device)*tdelays_frac.half().to(device)) + (dedisp_img_low.to(device)*(1 - tdelays_frac.half().to(device)))).to("cpu")
            else:
                dedisp_timeseries_all = (((dedisp_img_hi.to(device)*tdelays_frac.half().to(device)) + (dedisp_img_low.to(device)*(1 - tdelays_frac.half().to(device)))).sum(3)).to("cpu").float()
        elif torch.cuda.is_available():
            if device.index == 0: device2 = torch.device(1)
            else: device2 = torch.device(0)
            print("MOVING TO DEVICE 2," + str(device2),file=fout)
            
            #if device != None and device.type == 'cuda':
            #    if device.index == 0: device = torch.device(1)
            #    else: device = torch.device(0)
            subgridsize = gridsize//DMbatches
            image_tesseract_point_DM = image_tesseract_point.unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))
            print("IMG:"+str(image_tesseract_point_DM)+ "," + str(torch.sum(torch.isinf(image_tesseract_point_DM))),file=fout)
            print("IMG HALF:"+str(image_tesseract_point_DM.half()) + "," + str(torch.sum(torch.isinf(image_tesseract_point_DM.half()))),file=fout)
            print("JUST REAL SAMPLES:" +str(image_tesseract_point_DM[:,:,11:13,:,0])+ "," + str(torch.sum(torch.isinf(image_tesseract_point_DM[:,:,11:13,:,0]))),file=fout)
            #print("IMG:"+str(image_tesseract_point_DM),file=fout)
            dedisp_img_hi = torch.zeros(image_tesseract_point_DM.shape)#.unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))
            dedisp_img_low = torch.zeros(image_tesseract_point_DM.shape)#.unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))
            if keepfreqaxis:
                dedisp_img = torch.zeros(image_tesseract_point_DM.shape)
            else:
                dedisp_timeseries_all = torch.zeros((gridsize,gridsize,nsamps,len(DM_trials)))
            tdelays_frac = tdelays_frac.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(subgridsize,subgridsize,nsamps,-1,-1).to(device2)
            corr_shifts_all_hi_i = corr_shifts_all_hi.long().unsqueeze(0).unsqueeze(0).expand(subgridsize,subgridsize,-1,-1,-1).to(device2)
            corr_shifts_all_low_i = corr_shifts_all_low.long().unsqueeze(0).unsqueeze(0).expand(subgridsize,subgridsize,-1,-1,-1).to(device2)
            dedisp_img_i = torch.zeros((subgridsize,subgridsize,nsamps,nchans,len(DM_trials))).double().to(device2)
            
            for j in range(DMbatches):
                for i in range(DMbatches):
                    image_tesseract_point_DM_i = image_tesseract_point_DM.double()[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:,:].to(device2)
                
                    torch.gather(input=image_tesseract_point_DM_i,dim=2,index=corr_shifts_all_hi_i,out=dedisp_img_i)
                    if keepfreqaxis:
                        dedisp_img[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:,:] += (dedisp_img_i*tdelays_frac).to("cpu").float()
                    else:
                        dedisp_timeseries_all[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:] += (dedisp_img_i*tdelays_frac).sum(3).to("cpu").float()
                    torch.gather(input=image_tesseract_point_DM_i,dim=2,index=corr_shifts_all_low_i,out=dedisp_img_i)
                    if keepfreqaxis:
                        dedisp_img[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:,:] += (dedisp_img_i*(1-tdelays_frac)).to("cpu").float()
                    else:
                        dedisp_timeseries_all[j*subgridsize:(j+1)*subgridsize,i*subgridsize:(i+1)*subgridsize,:,:] += (dedisp_img_i*(1-tdelays_frac)).sum(3).to("cpu").float()


            
            torch.cuda.empty_cache()
            #del corr_shifts_all_hi_i
            #del corr_shifts_all_low_i
            """
            del dedisp_img_i
            del tdelays_frac
            del image_tesseract_point_DM_i
            """
        else:
            tdelays_frac = tdelays_frac.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],nsamps,-1,-1)
            corr_shifts_all_hi = corr_shifts_all_hi.long().unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,-1,-1)
            corr_shifts_all_low = corr_shifts_all_low.long().unsqueeze(0).unsqueeze(0).expand(image_tesseract_point.shape[0],image_tesseract_point.shape[1],-1,-1,-1)
            image_tesseract_point_DM = image_tesseract_point.unsqueeze(4).expand(-1,-1,-1,-1,len(DM_trials))
            print("IMG:"+str(image_tesseract_point_DM),file=fout)
            dedisp_img_hi = (torch.gather(image_tesseract_point_DM.double().to("cpu"),dim=2,index=torch.clip(corr_shifts_all_hi.to("cpu"),min=0,max=nsamps-1)))
            dedisp_img_low = (torch.gather(image_tesseract_point_DM.double().to("cpu"),dim=2,index=torch.clip(corr_shifts_all_low.to("cpu"),min=0,max=nsamps-1)))
        
            #dedisp_img_hi = (dedisp_img_hi.to(device)*tdelays_frac.half().to(device)).to("cpu")
            #dedisp_img_low = (dedisp_img_low.to(device)*(1 - tdelays_frac.half().to(device))).to("cpu")
            if keepfreqaxis:
                dedisp_img = ((dedisp_img_hi.to(device)*tdelays_frac.double().to(device)) + (dedisp_img_low.to(device)*(1 - tdelays_frac.double().to(device)))).to("cpu")
            else:
                dedisp_timeseries_all = ((dedisp_img_hi.to(device)*tdelays_frac.double().to(device)) + (dedisp_img_low.to(device)*(1 - tdelays_frac.double().to(device)))).sum(3).to("cpu")
        

        #dedisp_img = (torch.gather(image_tesseract_point_DM.to(device),dim=2,index=corr_shifts_all_hi.to(device))*(tdelays_frac.to(device))).to("cpu") + (torch.gather(image_tesseract_point_DM.to(device),dim=2,index=corr_shifts_all_low.to(device))*(1-tdelays_frac.to(device))).to("cpu")
        #dedisp_timeseries_all = (dedisp_img*(~torch.logical_or(corr_shifts_all_hi.to("cpu") < 0, corr_shifts_all_low.to("cpu") < 0))).sum(3)
        print("SHAPES:" + str(tdelays_idx_hi.to("cpu").shape),file=fout)
        #dedisp_timeseries_all = (dedisp_img.sum(3)).to("cpu").float()#*(~torch.logical_or(tdelays_idx_hi.unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cpu")>=nsamps,tdelays_idx_low.unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cpu")>=nsamps))).sum(3)
        torch.cuda.empty_cache()

        #del idxs_all
        """
        del corr_shifts_all_hi
        del corr_shifts_all_low
        del image_tesseract_point_DM
        del dedisp_img_hi
        del dedisp_img_low
        #del tdelays_hi
        #del tdelays_low
        #del tdelays_frac
        #del freq_axis
        del dedisp_img
        #dedisp_img = dedisp_img.to("cpu")
        #print("TDEL_FRAC:" + str(tdelays_frac),file=fout)dd
        """
        torch.cuda.empty_cache()


    else:
        if keepfreqaxis:
            dedisp_img = np.zeros((image_tesseract_point.shape[0],image_tesseract_point.shape[1],image_tesseract_point.shape[2],image_tesseract_point.shape[3],len(DM_trials)))
        else:
            dedisp_timeseries_all = np.zeros((image_tesseract_point.shape[0],image_tesseract_point.shape[1],image_tesseract_point.shape[2],len(DM_trials)))
        
        for j in range(len(DM_trials)):
            DM = DM_trials[j]
                
            #get delay axis
            neg_flag = DM < 0
            DM = np.abs(DM)

            #Delays
            tdelays = DM*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
            tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
            tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
            tdelays_frac = tdelays/tsamp - tdelays_idx_low
            print("Trial DM: " + str(DM) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout,end="")
            nchans = len(freq_axis)
            nsamps = image_tesseract_point.shape[-2]

            #shift each channel
            for k in range(nchans):
                #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac);
                if neg_flag:
                    padshape = tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-2) + [(tdelays_idx_low[k],0)])
                    arrlow =  np.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",constant_values=0)[:,:,:nsamps]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                else:
                    padshape = tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-2) + [(0,tdelays_idx_low[k])])
                    arrlow =  np.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",constant_values=0)[:,:,tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                print(padshape,file=fout)

                if neg_flag:
                    padshape = tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-2) + [(tdelays_idx_hi[k],0)])
                    arrhi =  np.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",constant_values=0)[:,:,:nsamps]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                else:
                    padshape = tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-2) + [(0,tdelays_idx_hi[k])])
                    arrhi =  np.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",constant_values=0)[:,:,tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

                print(padshape,file=fout)
                
                if keepfreqaxis:
                    dedisp_img[:,:,:,k,j] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])
                else:
                    dedisp_timeseries_all[:,:,:,j] += arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])

    #save frame
    if append_frame:
        save_last_frame(image_tesseract_point)
        print("Writing to last_frame.npy",file=fout)

    if keepfreqaxis:
        print("DEDISPERSED_ALL:" + str(dedisp_img),file=fout)
    else:
        print("DEDISPERSED: " + str(dedisp_timeseries_all),file=fout)
    print("Done!",file=fout)
    if output_file != "":
        fout.close()
    if append_frame:
        if keepfreqaxis:
            return dedisp_img[:,:,:truensamps,:,:]
        else:
            return dedisp_timeseries_all[:,:,:truensamps,:]
    else:
        if keepfreqaxis:
            return dedisp_img
        else:
            return dedisp_timeseries_all



#Consolidated final search functions
def run_search_CPU(image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,freq_axis=freq_axis,
                   DM_trials=DM_trials,widthtrials=widthtrials,tsamp=tsamp,SNRthresh=SNRthresh,plot=False,
                   off=10,PSF=default_PSF,offpnoise=0.3,verbose=False,output_file="",noiseth=0.9,canddict=dict(),usefft=False,
                   multithreading=False,nrows=1,ncols=1,space_filter=True,raidx_offset=0,decidx_offset=0,dm_offset=0,
                   threadDM=False,samenoise=False,cuda=False,exportmaps=False,kernel_size=len(RA_axis),append_frame=True,DMbatches=1,SNRbatches=1,usejax=True,RA_cutoff=default_cutoff,applySNthresh=True,slow=False):
    """
    This function takes an image cube of shape npixels x npixels x nchannels x ntimes and runs a dedispersion search that returns
    a list of candidates' DM, pulse width, RA, declination, and time of arrival(?)
    """
    t1 = time.time()
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    printprefix = "[" + str(raidx_offset) + str(decidx_offset) + str(dm_offset) + "]"

    print("Using device: " + str(device),file=fout)



    #get axis sizes
    gridsize_RA = len(RA_axis)
    gridsize_DEC = len(DEC_axis)
    gridsize = gridsize_RA
    nsamps = len(time_axis)
    nchans = len(freq_axis)

    print("Time for setup: " + str(time.time()-t1) + " s",file=fout)
    print("PRE-FILTER SHAPE: " + str(image_tesseract.shape) + ","  + str(PSF.shape),file=fout)
    if space_filter:
        t1 = time.time()
        if not slow:
            assert(gridsize_RA == gridsize_DEC)

        print(printprefix +"Spatial matched filtering with DSA PSF...",file=fout)
        if usefft:
            print(printprefix +"Using 2D FFT method...",file=fout)
        image_tesseract_filtered = matched_filter_space(image_tesseract,PSF,kernel_size=kernel_size,usefft=usefft,device=device)
    else:
        image_tesseract_filtered = image_tesseract
    #REVISED IMPLEMENTATION: DO DEDISP AND BOXCAR TOGETHER SO WE DON'T HAVE TO KEEP MOVING TO/FROM GPU
    total_noise=None

    #use the concurrent futures package to search sub-images separately; we have to do this AFTER the spatial matched filter so that the PSF structure is suppressed
    if multithreading:
        #initialize a pool of processes for concurent execution
        if threadDM:maxProcesses = nrows*ncols*len(DM_trials)
        else: maxProcesses = nrows*ncols
        executor = ThreadPoolExecutor(maxProcesses)

        #submit a search task to the process pool for each subimage
        assert(gridsize_DEC%nrows == 0) #gridsize must be divisible by number of rows and cols
        assert(gridsize_RA%ncols == 0)
        #nrows = ncols = int(np.sqrt(maxProcesses))
        gridsize_DEC_i = int(gridsize_DEC//nrows)
        gridsize_RA_i = int(gridsize_RA//ncols)
        candidxs = []
        cands = []
        canddict['ra_idxs'] = np.array([],dtype=int)
        canddict['dec_idxs'] = np.array([],dtype=int)
        canddict['wid_idxs'] = np.array([],dtype=int)
        canddict['dm_idxs'] = np.array([],dtype=int)
        canddict['ras'] = np.array([])
        canddict['decs'] = np.array([])
        canddict['wids'] = np.array([])
        canddict['dms'] = np.array([])
        canddict['snrs'] = np.array([])

        nDMtrials = len(DM_trials)
        nwidthtrials = len(widthtrials)
        image_tesseract_binned = np.zeros((gridsize_DEC,gridsize_RA,nwidthtrials,nDMtrials))

        print("Multi-processing with " + str(nrows) + " Rows, " + str(ncols) + " Cols",file=fout)
        task_list = []
        for i in range(nrows):#range(maxProcesses):
            for j in range(ncols):
                #define RA and DEC axes and sub-image
                row = i#int(i%np.sqrt(maxProcesses))
                col = j#int(i//np.sqrt(maxProcesses))
                image_tesseract_filtered_i = image_tesseract_filtered[row*gridsize_DEC_i:(row+1)*gridsize_DEC_i,col*gridsize_RA_i:(col+1)*gridsize_RA_i,:,:]
                RA_axis_i = RA_axis[col*gridsize_RA_i:(col+1)*gridsize_RA_i]
                DEC_axis_i = DEC_axis[row*gridsize_DEC_i:(row+1)*gridsize_DEC_i]
                print("---> Subimage " + str(i) + " (row=" + str(row) + ",col=" + str(col) + "), shape=" + str(image_tesseract_filtered_i.shape),file=fout)
                print(output_file[:-4] + "_thread_row" + str(i) + "_col" + str(j) + ".txt",file=fout)

                if threadDM:
                    for k in range(nDMtrials):
                        #define sub-range of DM trials
                        DM_trials_i = DM_trials[k:k+1]


                        #make new thread to search sub-image
                        task_list.append(executor.submit(run_search_new,
                                    image_tesseract_filtered_i,
                                    RA_axis_i,
                                    DEC_axis_i,
                                    time_axis,
                                    freq_axis,
                                    DM_trials_i,widthtrials,tsamp,SNRthresh,plot,
                                    off,PSF,offpnoise,verbose,output_file,noiseth,dict(),usefft,
                                    False,1,1,False,col*gridsize_RA_i,row*gridsize_DEC_i,k,False,samenoise,cuda))

                else:
                    #make new thread to search sub-image
                    task_list.append(executor.submit(run_search_new,
                                    image_tesseract_filtered_i,
                                    RA_axis_i,
                                    DEC_axis_i,
                                    time_axis,
                                    freq_axis,
                                    DM_trials,widthtrials,tsamp,SNRthresh,plot,
                                    off,PSF,offpnoise,verbose,output_file,noiseth,dict(),usefft,
                                    False,1,1,False,col*gridsize_RA_i,row*gridsize_DEC_i,0,False,samenoise,cuda))


        for future in as_completed(task_list):
            print("---> Result " + str(i) + ":",file=fout)
            candidxs_i,cands_i,image_tesseract_binned_i,image_tesseract_filtered_i,canddict_i,DM_trials_i,raidx_offset_i,decidx_offset_i,dm_offset_i = future.result()
            #if threadDM: subDMidx = list(DM_trials).index(DM_trials_i[0])#np.argmin(np.abs(DM_trials_i[0] - DM_trials))

            if applySNthresh:
                #save the binned image and candidates
                candidxs = list(candidxs) + list(candidxs_i)
                cands = list(cands) + list(cands_i)
                if threadDM: image_tesseract_binned[decidx_offset_i:decidx_offset_i + gridsize_DEC_i,raidx_offset_i:raidx_offset_i + gridsize_RA_i,:,dm_offset_i:dm_offset_i+1] = image_tesseract_binned_i
                else: image_tesseract_binned[decidx_offset_i:decidx_offset_i + gridsize_DEC_i,raidx_offset_i:raidx_offset_i + gridsize_RA_i,:,:] = image_tesseract_binned_i

                for k in canddict_i.keys():
                    canddict[k] = np.concatenate([canddict[k],canddict_i[k]])

        #make a dictionary for easy plotting of results
        if applySNthresh:
            ncands = len(cands)

    else: #proceed normally

        t1 = time.time()
        #dedisperse --> gridsize x gridsize x time x DM
        nDMtrials = len(DM_trials)
        print(printprefix +"Starting dedispersion with " + str(nDMtrials) + " trials...",file=fout)
        #image_tesseract_dedisp = np.zeros((gridsize_DEC,gridsize_RA,nsamps,nDMtrials)) 


        if usingGPU:
            if multithreading:
                image_tesseract_dedisp = dedisperse_allDM(torch.from_numpy(image_tesseract_filtered),DM_trials=DM_trials,tsamp=tsamp,freq_axis=freq_axis,output_file=output_file,device=device,append_frame=append_frame,DMbatches=DMbatches,maxProcesses=maxProcesses).numpy()
            elif usejax:
                image_tesseract_dedisp = dedisperse_allDM(image_tesseract_filtered,DM_trials=DM_trials,tsamp=tsamp,freq_axis=freq_axis,output_file=output_file,device=device,append_frame=append_frame,DMbatches=DMbatches,usejax=usejax)
            else:
                image_tesseract_dedisp = dedisperse_allDM(torch.from_numpy(image_tesseract_filtered),DM_trials=DM_trials,tsamp=tsamp,freq_axis=freq_axis,output_file=output_file,device=device,append_frame=append_frame,DMbatches=DMbatches,usejax=usejax).numpy()
        else:
            if multithreading:
                image_tesseract_dedisp = dedisperse_allDM(image_tesseract_filtered,DM_trials=DM_trials,tsamp=tsamp,freq_axis=freq_axis,output_file=output_file,device=device,append_frame=append_frame,DMbatches=DMbatches,maxProcesses=maxProcesses)
            else:
                image_tesseract_dedisp = dedisperse_allDM(image_tesseract_filtered,DM_trials=DM_trials,tsamp=tsamp,freq_axis=freq_axis,output_file=output_file,device=device,append_frame=append_frame,DMbatches=DMbatches)
        #print(image_tesseract_dedisp.shape)
        print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_dedisp))),file=fout)

        print(printprefix +"Done!",file=fout)
        print("Time for dedispersion: " + str(time.time()-t1) + " s",file=fout)

        t1 = time.time()
        #boxcar filter and get snr using rolled PSF --> gridsize x gridsize x width x DM (x TOA?)
        nwidthtrials = len(widthtrials)
        image_tesseract_binned = np.zeros((gridsize_DEC,gridsize_RA,nwidthtrials,nDMtrials)) #stores output array as S/N for each dedispersion and width trial for every pixel

        print(printprefix +"Starting boxcar filtering with " + str(nwidthtrials) + " trials...",file=fout)


        if usingGPU:
            image_tesseract_binned = snr_vs_RA_DEC_allDMW(torch.from_numpy(image_tesseract_dedisp),DM_trials,widthtrials,noiseth=noiseth,output_file=output_file,samenoise=samenoise,device=device,exportmaps=exportmaps,usefft=usefft,batches=SNRbatches,usejax=usejax).numpy()
        else:
            image_tesseract_binned = snr_vs_RA_DEC_allDMW(image_tesseract_dedisp,DM_trials,widthtrials,noiseth=noiseth,output_file=output_file,samenoise=samenoise,device=device,exportmaps=exportmaps,usefft=usefft,batches=SNRbatches)
        print(printprefix +"Done!",file=fout)
        print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_binned))),file=fout)
        print("Time for boxcar filter: " + str(time.time()-t1) + " s",file=fout)

    
        t1 = time.time()
        if applySNthresh:
            print(printprefix +"Searching for candidates with S/N > " + str(SNRthresh) + "...",file=fout)
            #find candidates above SNR threshold
            #condition = (image_tesseract_binned>=SNRthresh).flatten()
            #ncands = np.sum(condition)
            #canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs=np.unravel_index(np.arange(gridsize_DEC*gridsize_RA*nDMtrials*nwidthtrials)[condition],(gridsize_DEC,gridsize_RA,nwidthtrials,nDMtrials))#[1].shape

            canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs = np.nonzero(image_tesseract_binned>=SNRthresh)
            ncands = len(canddec_idxs)

            canddecs = DEC_axis[canddec_idxs]
            candras = RA_axis[candra_idxs]
            candwids = widthtrials[candwid_idxs]
            canddms = DM_trials[canddm_idxs]
            candsnrs = image_tesseract_binned[canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs]#.flatten()[condition]

            candidxs = [(raidx_offset + candra_idxs[i],decidx_offset + canddec_idxs[i],candwid_idxs[i],dm_offset + canddm_idxs[i],candsnrs[i]) for i in range(ncands)]
            cands = [(candras[i],canddecs[i],candwids[i],canddms[i],candsnrs[i]) for i in range(ncands)]

            #make a dictionary for easy plotting of results
            canddict['ra_idxs'] = copy.deepcopy(candra_idxs + raidx_offset)
            canddict['dec_idxs'] = copy.deepcopy(canddec_idxs + decidx_offset)
            canddict['wid_idxs'] = copy.deepcopy(candwid_idxs)
            canddict['dm_idxs'] = copy.deepcopy(canddm_idxs + dm_offset)
            canddict['ras'] = copy.deepcopy(candras)
            canddict['decs'] = copy.deepcopy(canddecs)
            canddict['wids'] = copy.deepcopy(candwids)
            canddict['dms'] = copy.deepcopy(canddms)
            canddict['snrs'] = copy.deepcopy(candsnrs)
            print("Time for sorting candidates: " + str(time.time()-t1) + " s",file=fout)
    if applySNthresh:
        print(printprefix +"Done! Found " + str(ncands) + " candidates",file=fout)
    if output_file != "":
        fout.close()
    if applySNthresh:
        if append_frame:
            return candidxs,cands,image_tesseract_binned,image_tesseract_filtered[:,:,-truensamps:,:],canddict,DM_trials,raidx_offset,decidx_offset,dm_offset,total_noise
        else:
            return candidxs,cands,image_tesseract_binned,image_tesseract_filtered,canddict,DM_trials,raidx_offset,decidx_offset,dm_offset,total_noise
    else:
        if append_frame:
            return TOAs,image_tesseract_binned,image_tesseract_filtered[:,:,-truensamps:,:],DM_trials,raidx_offset,decidx_offset,dm_offset,total_noise
        else:
            return TOAs,image_tesseract_binned,image_tesseract_filtered,DM_trials,raidx_offset,decidx_offset,dm_offset,total_noise



"""
'ATTACH' MODE, ADD DICTIONARY WITH SECONDARY SEARCH PARAMS:
    key = isot_slow/imgdiff
    image_tesseract
    RA_axis,
    DEC_axis,
    time_axis
    tsamp
    append_frame
    RA_cutoff
    slow
    imgdiff
"""

def run_search_GPU(image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,freq_axis=freq_axis,
                   DM_trials=DM_trials,widthtrials=widthtrials,tsamp=tsamp,SNRthresh=SNRthresh,plot=False,
                   off=10,PSF=default_PSF,offpnoise=0.3,verbose=False,output_file="",noiseth=0.9,canddict=dict(),usefft=False,
                   multithreading=False,nrows=1,ncols=1,space_filter=True,
                   threadDM=False,samenoise=False,cuda=False,exportmaps=False,kernel_size=len(RA_axis),append_frame=True,DMbatches=1,SNRbatches=1,usejax=True,RA_cutoff=default_cutoff,applySNthresh=True,slow=False,imgdiff=False,attach=dict(),completeness=False,forfeit=False):
    """
    This function takes an image cube of shape npixels x npixels x nchannels x ntimes and runs a dedispersion search that returns
    a list of candidates' DM, pulse width, RA, declination, and time of arrival(?)
    """


    t1 = time.time()
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #get axis sizes
    gridsize_RA = len(RA_axis)
    gridsize_DEC = len(DEC_axis)
    gridsize = gridsize_RA
    nsamps = len(time_axis)
    nchans = len(freq_axis)
    if slow:
        imgdiff = False
    elif imgdiff:
        assert(image_tesseract.shape[-1]==1)
        print("Image differencing mode")
        append_frame=False
    
    for k in attach.keys():
        attach[k]['gridsize_RA'] = len(attach[k]['RA_axis'])
        attach[k]['gridsize_DEC'] = len(attach[k]['DEC_axis'])
        attach[k]['gridsize'] = attach[k]['gridsize_RA']
        attach[k]['nsamps'] = len(attach[k]['time_axis'])
        #attach[k]['nchans'] = len(attach[k]['freq_axis'])
    
    
        if attach[k]['slow']:
            attach[k]['imgdiff'] = False
        elif attach[k]['imgdiff']:
            assert(attach[k]['image_tesseract'].shape[-1]==1)
            print("Image differencing mode")
            attach[k]['append_frame']=False

    device = torch.device(random.choice(np.arange(torch.cuda.device_count(),dtype=int)) if torch.cuda.is_available() else "cpu")
    usingGPU = device.type == "cuda"
    if not usingGPU:
        print("GPU unavailable")
        return None
    print("Using device: " + str(device),file=fout)

    #REVISED IMPLEMENTATION: DO DEDISP AND BOXCAR TOGETHER SO WE DON'T HAVE TO KEEP MOVING TO/FROM GPU
    total_noise=None
    t1 = time.time()
    nwidths,ndms = len(widthtrials),len(DM_trials)
    #get data from previous timeframe
    global last_frame_slow
    global last_frame
    if append_frame:
        #if slow:
        #    global last_frame_slow
        #else:
        #    global last_frame
        #global last_frame
        print("OLD SHAPE:",image_tesseract.shape,file=fout)

        truensamps = image_tesseract.shape[2]
        if slow:
            if RA_cutoff>0:
                image_tesseract_filtered_cut = np.concatenate([last_frame_slow[:,:-RA_cutoff,:,:],image_tesseract[:,RA_cutoff:,:,:]],axis=2)
            else:
                image_tesseract_filtered_cut = np.concatenate([last_frame_slow,image_tesseract],axis=2)
        else:
            if RA_cutoff>0:
                image_tesseract_filtered_cut = np.concatenate([last_frame[:,:-RA_cutoff,:,:],image_tesseract[:,RA_cutoff:,:,:]],axis=2)
            else:
                image_tesseract_filtered_cut = np.concatenate([last_frame,image_tesseract],axis=2)
        if PSF.shape[1]>= image_tesseract_filtered_cut.shape[1]:
            #first trim to equal dimensions as image
            PSF = PSF[int((PSF.shape[0]-image_tesseract.shape[0])//2):int((PSF.shape[0]-image_tesseract.shape[0])//2)+image_tesseract.shape[0],
                      int((PSF.shape[1]-image_tesseract.shape[1])//2):int((PSF.shape[1]-image_tesseract.shape[1])//2)+image_tesseract.shape[1],:,:]
            #then apply RA cutoff
            if RA_cutoff>0:
                PSF = PSF[:,int(RA_cutoff//2):-(RA_cutoff - int(RA_cutoff//2)),:,:]
        nsamps = image_tesseract.shape[2]
        if slow:
            corr_shifts_all = corr_shifts_all_append_slow
            tdelays_frac = tdelays_frac_append_slow
        else:
            corr_shifts_all = corr_shifts_all_append
            tdelays_frac = tdelays_frac_append
        print("NEW SHAPE:",image_tesseract_filtered_cut.shape,file=fout)
        print("MAXSHIFT:",maxshift,file=fout)
        print("Appending data from previous timeframe, new shape: " + str(image_tesseract_filtered_cut.shape),file=fout)

        #save frame
        if slow:
            last_frame_slow = image_tesseract[:,:,-maxshift_slow:,:]
        else:
            last_frame = image_tesseract[:,:,-maxshift:,:]
        
        if RA_cutoff>0:
            RA_axis = RA_axis[RA_cutoff:]
            gridsize_RA = len(RA_axis)
    else:
        if PSF.shape[1]>= image_tesseract.shape[1]:
            #first trim to equal dimensions as image
            PSF = PSF[int((PSF.shape[0]-image_tesseract.shape[0])//2):int((PSF.shape[0]-image_tesseract.shape[0])//2)+image_tesseract.shape[0],
                      int((PSF.shape[1]-image_tesseract.shape[1])//2):int((PSF.shape[1]-image_tesseract.shape[1])//2)+image_tesseract.shape[1],:,:]
        if slow:
            corr_shifts_all = corr_shifts_all_no_append_slow
            tdelays_frac = tdelays_frac_no_append_slow
        else:
            corr_shifts_all = corr_shifts_all_no_append
            tdelays_frac = tdelays_frac_no_append
        truensamps = nsamps = image_tesseract.shape[2]
        image_tesseract_filtered_cut=image_tesseract
    assert(PSF.shape[0]<=image_tesseract_filtered_cut.shape[0])
    assert(PSF.shape[1]<=image_tesseract_filtered_cut.shape[1])
    #subgrid
    subgridsize_RA = gridsize_RA//DMbatches
    subgridsize_DEC = gridsize_DEC//DMbatches
    
    #noise prep
    #total_noise = np.zeros((nwidths,ndms))

    if imgdiff:
        prev_noise,prev_noise_N = np.zeros((len(widthtrials),len(DM_trials))),0
    else:
        prev_noise,prev_noise_N = copy.deepcopy(current_noise) #noise_update_all(None,gridsize_RA,gridsize_DEC,DM_trials,widthtrials,readonly=True)
        if slow:
            prev_noise /= np.sqrt(config.bin_slow)

    for k in attach.keys():
        if attach[k]['append_frame']:
            #if attach[k]['slow']:
            #    global last_frame_slow
            #else:
            #    global last_frame
            #global last_frame
            print("OLD SHAPE:",attach[k]['image_tesseract'].shape,file=fout)

            attach[k]['truensamps'] = attach[k]['image_tesseract'].shape[2]
            if attach[k]['slow']:
                if attach[k]['RA_cutoff']>0:
                    attach[k]['image_tesseract_filtered_cut'] = np.concatenate([last_frame_slow[:,:-attach[k]['RA_cutoff'],:,:],attach[k]['image_tesseract'][:,attach[k]['RA_cutoff']:,:,:]],axis=2)
                else:
                    attach[k]['image_tesseract_filtered_cut'] = np.concatenate([last_frame_slow,attach[k]['image_tesseract']],axis=2)
            else:
                if attach[k]['RA_cutoff']>0:
                    attach[k]['image_tesseract_filtered_cut'] = np.concatenate([last_frame[:,:-attach[k]['RA_cutoff'],:,:],attach[k]['image_tesseract'][:,attach[k]['RA_cutoff']:,:,:]],axis=2)
                else:
                    attach[k]['image_tesseract_filtered_cut'] = np.concatenate([last_frame,attach[k]['image_tesseract']],axis=2)
            if attach[k]['PSF'].shape[1]>= attach[k]['image_tesseract_filtered_cut'].shape[1]:
                #first trim to equal dimensions as image
                attach[k]['PSF'] = attach[k]['PSF'][int((attach[k]['PSF'].shape[0]-attach[k]['image_tesseract'].shape[0])//2):int((attach[k]['PSF'].shape[0]-attach[k]['image_tesseract'].shape[0])//2)+attach[k]['image_tesseract'].shape[0],
                      int((attach[k]['PSF'].shape[1]-attach[k]['image_tesseract'].shape[1])//2):int((attach[k]['PSF'].shape[1]-attach[k]['image_tesseract'].shape[1])//2)+attach[k]['image_tesseract'].shape[1],:,:]
                #then apply RA cutoff
                if attach[k]['RA_cutoff']>0:
                    attach[k]['PSF'] = attach[k]['PSF'][:,int(attach[k]['RA_cutoff']//2):-(attach[k]['RA_cutoff'] - int(attach[k]['RA_cutoff']//2)),:,:]
            attach[k]['nsamps'] = attach[k]['image_tesseract'].shape[2]
            if attach[k]['slow']:
                attach[k]['corr_shifts_all'] = corr_shifts_all_append_slow
                attach[k]['tdelays_frac'] = tdelays_frac_append_slow
            else:
                attach[k]['corr_shifts_all'] = corr_shifts_all_append
                attach[k]['tdelays_frac'] = tdelays_frac_append
            print("NEW SHAPE:",attach[k]['image_tesseract_filtered_cut'].shape,file=fout)
            #print("MAXSHIFT:",attach[k]['maxshift'],file=fout)
            print("Appending data from previous timeframe, new shape: " + str(attach[k]['image_tesseract_filtered_cut'].shape),file=fout)

            #save frame
            if attach[k]['slow']:
                last_frame_slow = attach[k]['image_tesseract'][:,:,-maxshift_slow:,:]
            else:
                last_frame = attach[k]['image_tesseract'][:,:,-maxshift:,:]

            if attach[k]['RA_cutoff']>0:
                attach[k]['RA_axis'] = attach[k]['RA_axis'][attach[k]['RA_cutoff']:]
                attach[k]['gridsize_RA'] = len(attach[k]['RA_axis'])
        else:
            if attach[k]['PSF'].shape[1]>= attach[k]['image_tesseract'].shape[1]:
                #first trim to equal dimensions as image
                attach[k]['PSF'] = attach[k]['PSF'][int((attach[k]['PSF'].shape[0]-attach[k]['image_tesseract'].shape[0])//2):int((attach[k]['PSF'].shape[0]-attach[k]['image_tesseract'].shape[0])//2)+attach[k]['image_tesseract'].shape[0],
                      int((attach[k]['PSF'].shape[1]-attach[k]['image_tesseract'].shape[1])//2):int((attach[k]['PSF'].shape[1]-attach[k]['image_tesseract'].shape[1])//2)+attach[k]['image_tesseract'].shape[1],:,:]
            if attach[k]['slow']:
                attach[k]['corr_shifts_all'] = corr_shifts_all_no_append_slow
                attach[k]['tdelays_frac'] = tdelays_frac_no_append_slow
            else:
                attach[k]['corr_shifts_all'] = corr_shifts_all_no_append
                attach[k]['tdelays_frac'] = tdelays_frac_no_append
            attach[k]['truensamps'] = attach[k]['nsamps'] = attach[k]['image_tesseract'].shape[2]
            attach[k]['image_tesseract_filtered_cut']=attach[k]['image_tesseract']
        assert(attach[k]['PSF'].shape[0]<=attach[k]['image_tesseract_filtered_cut'].shape[0])
        assert(attach[k]['PSF'].shape[1]<=attach[k]['image_tesseract_filtered_cut'].shape[1])
        #subgrid
        attach[k]['subgridsize_RA'] = attach[k]['gridsize_RA']//DMbatches
        attach[k]['subgridsize_DEC'] = attach[k]['gridsize_DEC']//DMbatches

        #noise prep
        #total_noise = np.zeros((nwidths,ndms))

        if attach[k]['imgdiff']:
            attach[k]['prev_noise'],attach[k]['prev_noise_N'] = np.zeros((len(widthtrials),len(DM_trials))),0
        else:
            attach[k]['prev_noise'],attach[k]['prev_noise_N'] = copy.deepcopy(current_noise) #noise_update_all(None,gridsize_RA,gridsize_DEC,DM_trials,widthtrials,readonly=True)
            if attach[k]['slow']:
                attach[k]['prev_noise'] /= np.sqrt(config.bin_slow)



    print("Time for Prep" + str(time.time()-t1),file=fout)
    t1 = time.time()
    print("NOISENUM:" + str(prev_noise_N),file=fout)
    print("INPUT NOISE:" + str(prev_noise),file=fout)


    #jaxdev = random.choice(np.arange(len(jax.devices()),dtype=int))
    global jaxdev
    usedev = ((jaxdev + 1)%2 if ((not forfeit) and (slow or imgdiff)) else jaxdev)
    if forfeit or (not (slow or imgdiff)):
        jaxdev += 1
        jaxdev %= 2
    print("DM TRIALS AND WIDTH TRIALS:" + str(corr_shifts_all.shape) + str(tdelays_frac.shape) + str(full_boxcar_filter.shape),file=fout)
    print(corr_shifts_all,file=fout)
    print(tdelays_frac,file=fout)
    print(full_boxcar_filter,file=fout)
    global jax_inuse
    if (not forfeit) and (slow or imgdiff):#realtime:
        print("WAITING HERE",file=fout)
        while jax_inuse[usedev]:
            continue
    jax_inuse[usedev] = True
    print(str("slow" if slow else "") + " JAX DEVICE",usedev,"AVAILABLE",file=fout)
    
    if completeness:
        if append_frame:
            outtup = jax_funcs.matched_filter_dedisp_snr_fft_jit_completeness(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                                                 #(default_PSF_gpu_0 if usedev==0 else default_PSF_gpu_1),
                                                                 jax.device_put(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 #(corr_shifts_all_gpu_0 if usedev==0 else corr_shifts_all_gpu_1),
                                                                 jax.device_put(corr_shifts_all,jax.devices()[usedev]),
                                                                 #(tdelays_frac_gpu_0 if usedev==0 else tdelays_frac_gpu_1),
                                                                 jax.device_put(tdelays_frac,jax.devices()[usedev]),
                                                                 #(full_boxcar_filter_gpu_0 if usedev==0 else full_boxcar_filter_gpu_1),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(prev_noise[:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 prev_noise_N,noiseth)
        elif imgdiff:
            outtup = jax_funcs.img_diff_jit_no_append_completeness(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(full_boxcar_filter_imgdiff,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(prev_noise[:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 prev_noise_N,noiseth)
        else:
    
            outtup = jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append_completeness(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                                                 #(jax_funcs.PSF_1 if usedev==0 else jax_funcs.PSF_2),#
                                                                 jax.device_put(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(corr_shifts_all,jax.devices()[usedev]),
                                                                 jax.device_put(tdelays_frac,jax.devices()[usedev]),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(prev_noise[:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 prev_noise_N,noiseth)
    else:
        if append_frame:
            outtup = jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                                                 #(default_PSF_gpu_0 if usedev==0 else default_PSF_gpu_1),
                                                                 jax.device_put(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 #(corr_shifts_all_gpu_0 if usedev==0 else corr_shifts_all_gpu_1),
                                                                 jax.device_put(corr_shifts_all,jax.devices()[usedev]),
                                                                 #(tdelays_frac_gpu_0 if usedev==0 else tdelays_frac_gpu_1),
                                                                 jax.device_put(tdelays_frac,jax.devices()[usedev]),
                                                                 #(full_boxcar_filter_gpu_0 if usedev==0 else full_boxcar_filter_gpu_1),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(prev_noise[:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 prev_noise_N,noiseth)
        elif imgdiff:
            outtup = jax_funcs.img_diff_jit_no_append(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(full_boxcar_filter_imgdiff,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(prev_noise[:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 prev_noise_N,noiseth)
        else:

            outtup = jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[usedev]),
                                                                 #(jax_funcs.PSF_1 if usedev==0 else jax_funcs.PSF_2),#
                                                                 jax.device_put(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(corr_shifts_all,jax.devices()[usedev]),
                                                                 jax.device_put(tdelays_frac,jax.devices()[usedev]),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(prev_noise[:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 prev_noise_N,noiseth)
            
            
            
    image_tesseract_binned,total_noise,TOAs = np.array(outtup[0]),np.array(outtup[1])[:,np.newaxis].repeat(len(DM_trials),1),np.array(outtup[2])
    
    for k in attach.keys():
        if attach[k]['append_frame']:
            attach[k]['outtup'] = jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(attach[k]['image_tesseract_filtered_cut'],dtype=np.float32),jax.devices()[usedev]),
                                                                 #(default_PSF_gpu_0 if usedev==0 else default_PSF_gpu_1),
                                                                 jax.device_put(np.array(attach[k]['PSF'][:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(attach[k]['PSF'][:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 #(corr_shifts_all_gpu_0 if usedev==0 else corr_shifts_all_gpu_1),
                                                                 jax.device_put(attach[k]['corr_shifts_all'],jax.devices()[usedev]),
                                                                 #(tdelays_frac_gpu_0 if usedev==0 else tdelays_frac_gpu_1),
                                                                 jax.device_put(attach[k]['tdelays_frac'],jax.devices()[usedev]),
                                                                 #(full_boxcar_filter_gpu_0 if usedev==0 else full_boxcar_filter_gpu_1),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(attach[k]['prev_noise'][:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 attach[k]['prev_noise_N'],noiseth)
        elif attach[k]['imgdiff']:
            attach[k]['outtup'] = jax_funcs.img_diff_jit_no_append(jax.device_put(np.array(attach[k]['image_tesseract_filtered_cut'],dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(attach[k]['PSF'][:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(attach[k]['PSF'][:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(full_boxcar_filter_imgdiff,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(attach[k]['prev_noise'][:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 attach[k]['prev_noise_N'],noiseth)
        else:
            attach[k]['outtup'] = jax_funcs.matched_filter_dedisp_snr_fft_jit_no_append(jax.device_put(np.array(attach[k]['image_tesseract_filtered_cut'],dtype=np.float32),jax.devices()[usedev]),
                                                                 #(jax_funcs.PSF_1 if usedev==0 else jax_funcs.PSF_2),#
                                                                 jax.device_put(np.array(attach[k]['PSF'][:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(attach[k]['PSF'][:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[usedev]),
                                                                 jax.device_put(attach[k]['corr_shifts_all'],jax.devices()[usedev]),
                                                                 jax.device_put(attach[k]['tdelays_frac'],jax.devices()[usedev]),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[usedev]),
                                                                 jax.device_put(np.array(attach[k]['prev_noise'][:,0],dtype=noise_data_type),jax.devices()[usedev]),
                                                                 attach[k]['prev_noise_N'],noiseth)
        attach[k]['image_tesseract_binned'],attach[k]['total_noise'],attach[k]['TOAs'] = np.array(attach[k]['outtup'][0]),np.array(attach[k]['outtup'][1])[:,np.newaxis].repeat(len(DM_trials),1),np.array(attach[k]['outtup'][2])
    jax_inuse[usedev] = False
    print(str("slow" if slow else "") + "YIELDING JAX DEVICE",usedev,file=fout)
    print("Time for DM and SNR:" + str(time.time()-t1),file=fout)

    if output_file != "":
        fout.close()
    if append_frame:
        return TOAs,image_tesseract_binned,image_tesseract[:,:,-truensamps:,:],total_noise
    else:
        return TOAs,image_tesseract_binned,image_tesseract,total_noise



    


#CONTEXTSETUP = False
def search_task(searchlock,fullimg,SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,space_filter,kernel_size,exportmaps,savesearch,fprtest,fnrtest,append_frame,DMbatches,SNRbatches,usejax,noiseth,nocutoff,realtime,slow,imgdiff,attach_fullimg_slow=None,attach_fullimg_imgdiff=None,attach_mode=False,completeness=False,forfeit=False):
    searchlock.acquire()
    global current_noise
    printlog("CURRENT NOISE:"+str((current_noise[0][0,0],current_noise[1])),run_file)
    if forfeit:
        printlog("FORFEIT MODE",output_file=processfile)
    timing1 = time.time()
    #global CONTEXTSETUP
    #if not QSETUP and not CONTEXTSETUP:
    #    CONTEXTSETUP = True
    #    torch.multiprocessing.set_start_method("spawn")
    printlog("starting" + (" slow " if slow else " ") + "search process " + str(fullimg.img_id_isot) + "...",output_file=processfile,end='')



    if imgdiff:#slow or imgdiff:
        append_frame = False

    #define search params
    gridsize=fullimg.image_tesseract.shape[0]
    RA_axis = fullimg.RA_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
    DEC_axis= fullimg.DEC_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
    nsamps = fullimg.image_tesseract.shape[2]
    nchans = fullimg.image_tesseract.shape[3]
    if imgdiff:
        time_axis = np.arange(nsamps)*tsamp_imgdiff
    else:
        time_axis = np.arange(nsamps)*(tsamp_slow if slow else tsamp)
    global default_PSF
    global default_PSF_params
    global PSF_dict
    if not realtime:
        default_PSF,default_PSF_params = scPSF.manage_PSF(PSF_dict,kernel_size,fullimg.img_dec,default_PSF_params,default_PSF,nsamps=nsamps)
        """tmppsf = jax.device_put(jax_funcs.PSF_1,jax.devices("cpu")[0])
        tmppsf = jax.device_put(jax_funcs.PSF_2,jax.devices("cpu")[0])
        jax_funcs.PSF_1 = jax.device_put(np.array(default_PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(default_PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[0])
        jax_funcs.PSF_2 = jax.device_put(np.array(default_PSF[:,:,0:1,:].sum(3,keepdims=True)/np.sum(np.array(default_PSF[:,:,0:1,:].sum(3,keepdims=True))),dtype=np.float32),jax.devices()[1])"""

    global last_frame
    global last_frame_init_idx
    global last_frame_slow
    global last_frame_slow_init_idx
    printlog("LAST FRAME: " + str(last_frame_init_idx) + str(last_frame.shape) + str(last_frame) + "...",output_file=processfile,end='')
    if (not slow) and (last_frame.shape[:2] != fullimg.image_tesseract.shape[:2] or last_frame_init_idx==0):
        last_frame = get_last_frame()
        last_frame_init_idx += 1
        printlog("AFTER UPDATE LAST FRAME: " + str(last_frame_init_idx) + str(last_frame.shape) + str(last_frame) + "...",output_file=processfile,end='')
    elif (slow) and (last_frame_slow.shape[:2] != fullimg.image_tesseract.shape[:2] or last_frame_slow_init_idx==0):
        last_frame_slow = get_last_frame(slow=True)[:,-fullimg.image_tesseract.shape[1]:,:,:]
        last_frame_slow_init_idx += 1
        printlog("<SLOW>AFTER UPDATE LAST FRAME: " + str(last_frame_slow_init_idx) + str(last_frame_slow.shape) + str(last_frame_slow) + "...",output_file=processfile,end='')
    #printlog("AFTER UPDATE LAST FRAME: " + str(last_frame_init_idx) + str(last_frame.shape) + str(last_frame) + "...",output_file=processfile,end='')
    if cuda:
        attach = dict()
    if cuda and (attach_fullimg_slow is not None):
        attach['slow'] = dict()
        attach['slow']['append_frame'] = append_frame
        attach['slow']['slow'] =True
        attach['slow']['imgdiff'] = False
        #define search params
        attach['slow']['gridsize']=attach_fullimg_slow.image_tesseract.shape[0]
        attach['slow']['RA_axis'] = attach_fullimg_slow.RA_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
        attach['slow']['DEC_axis'] = attach_fullimg_slow.DEC_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
        attach['slow']['nsamps'] = attach_fullimg_slow.image_tesseract.shape[2]
        #attach['slow']['nchans'] = attach_fullimg_slow.image_tesseract.shape[3]
        attach['slow']['time_axis'] = np.arange(attach['slow']['nsamps'])*(tsamp_slow)
        attach['slow']['PSF'] = default_PSF
        if (last_frame_slow.shape != attach_fullimg_slow.image_tesseract.shape or last_frame_slow_init_idx==0):
            last_frame_slow = get_last_frame(slow=True)[:,-attach_fullimg_slow.image_tesseract.shape[1]:,:,:]
            last_frame_slow_init_idx += 1
            printlog("<SLOW>AFTER UPDATE LAST FRAME: " + str(last_frame_slow_init_idx) + str(last_frame_slow.shape) + str(last_frame_slow) + "...",output_file=processfile,end='')
    
        attach['slow']['image_tesseract'] = attach_fullimg_slow.image_tesseract
        attach['slow']['RA_cutoff'] = 0 if nocutoff else get_RA_cutoff(attach_fullimg_slow.img_dec,T=tsamp_slow*nsamps)
        attach['slow']['tsamp'] = tsamp_slow
    if cuda and (attach_fullimg_imgdiff is not None):
        
        attach['imgdiff'] = dict()
        attach['imgdiff']['append_frame'] = False
        attach['imgdiff']['slow'] =False
        attach['imgdiff']['imgdiff'] = True
        #define search params
        attach['imgdiff']['gridsize']=attach_fullimg_imgdiff.image_tesseract.shape[0]
        attach['imgdiff']['RA_axis'] = attach_fullimg_imgdiff.RA_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
        attach['imgdiff']['DEC_axis'] = attach_fullimg_imgdiff.DEC_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
        attach['imgdiff']['nsamps'] = attach_fullimg_imgdiff.image_tesseract.shape[2]
        #attach['imgdiff']['nchans'] = attach_fullimg_imgdiff.image_tesseract.shape[3]
        attach['imgdiff']['time_axis'] = np.arange(attach['imgdiff']['nsamps'])*(tsamp_imgdiff)
        attach['imgdiff']['PSF'] = default_PSF
        attach['imgdiff']['image_tesseract'] = attach_fullimg_imgdiff.image_tesseract
        attach['imgdiff']['RA_cutoff'] = 0 #if nocutoff else get_RA_cutoff(attach_fullimg_slow.img_dec,T=tsamp_slow*nsamps)
        attach['imgdiff']['tsamp'] = tsamp_imgdiff

    #print("starting process " + str(img_id) + "...")
    if cuda:
        TOAs,fullimg.image_tesseract_searched,fullimg.image_tesseract_binned,total_noise = run_search_GPU(fullimg.image_tesseract,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,canddict=dict(),usefft=usefft,multithreading=multithreading,nrows=nrows,ncols=ncols,output_file=output_file,threadDM=threadDM,samenoise=samenoise,cuda=cuda,space_filter=space_filter,kernel_size=kernel_size,exportmaps=exportmaps,append_frame=(False if imgdiff else append_frame),DMbatches=DMbatches,SNRbatches=SNRbatches,usejax=usejax,noiseth=noiseth,RA_cutoff=0 if nocutoff else get_RA_cutoff(fullimg.img_dec,T=(tsamp_slow if slow else tsamp)*nsamps,pixsize=np.abs(fullimg.RA_axis[1]-fullimg.RA_axis[0])),DM_trials=DM_trials,widthtrials=widthtrials,applySNthresh=False,slow=slow,imgdiff=imgdiff,attach=attach,completeness=completeness,forfeit=forfeit) 

    else:
        TOAs,fullimg.image_tesseract_searched,fullimg.image_tesseract_binned,tmp,tmp,tmp,tmp,total_noise = run_search_CPU(fullimg.image_tesseract,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,canddict=dict(),usefft=usefft,multithreading=multithreading,nrows=nrows,ncols=ncols,output_file=output_file,threadDM=threadDM,samenoise=samenoise,cuda=cuda,space_filter=space_filter,kernel_size=kernel_size,exportmaps=exportmaps,append_frame=(False if imgdiff else append_frame),DMbatches=DMbatches,SNRbatches=SNRbatches,usejax=usejax,noiseth=noiseth,RA_cutoff=0 if nocutoff else get_RA_cutoff(fullimg.img_dec,T=(tsamp_slow if slow else tsamp)*nsamps,pixsize=np.abs(fullimg.RA_axis[1]-fullimg.RA_axis[0])),DM_trials=DM_trials,widthtrials=widthtrials,applySNthresh=False,slow=slow,imgdiff=imgdiff,completeness=completeness,forfeit=forfeit)
    
    cands_found = np.nanmax(fullimg.image_tesseract_searched)>SNRthresh
    if cuda and (attach_fullimg_slow is not None):
        if attach['slow']['append_frame']:
            attach_fullimg_slow.image_tesseract_searched,attach_fullimg_slow.image_tesseract_binned = attach['slow']['image_tesseract_binned'],attach['slow']['image_tesseract'][:,:,-attach['slow']['truensamps']:,:]
        else:
            attach_fullimg_slow.image_tesseract_searched,attach_fullimg_slow.image_tesseract_binned = attach['slow']['image_tesseract_binned'],attach['slow']['image_tesseract']
        
        attach['slow']['cands_found'] = np.nanmax(attach_fullimg_slow.image_tesseract_searched)>SNRthresh
    if cuda and (attach_fullimg_imgdiff is not None):
        attach_fullimg_imgdiff.image_tesseract_searched,attach_fullimg_imgdiff.image_tesseract_binned = attach['imgdiff']['image_tesseract_binned'],attach['imgdiff']['image_tesseract']

        attach['imgdiff']['cands_found'] = np.nanmax(attach_fullimg_imgdiff.image_tesseract_searched)>SNRthresh
    """
    if not (slow or imgdiff):
        padby=(fullimg.image_tesseract_searched.shape[0]-fullimg.image_tesseract_searched.shape[1],0)
        fullimg.image_tesseract_searched = np.pad(fullimg.image_tesseract_searched,((0,0),padby,(0,0),(0,0)),constant_values=np.nan)
        TOAs = np.pad(TOAs,((0,0),padby,(0,0),(0,0)),constant_values=np.nan)
    elif slow:
        padby=(fullimg.image_tesseract.shape[1]-fullimg.image_tesseract_searched.shape[1],fullimg.image_tesseract.shape[0]-fullimg.image_tesseract.shape[1])
        padby2 = (padby[0],0)
        fullimg.image_tesseract_searched = np.pad(fullimg.image_tesseract_searched,((0,0),padby,(0,0),(0,0)),constant_values=np.nan)
        TOAs = np.pad(TOAs,((0,0),padby,(0,0),(0,0)),constant_values=np.nan)
    """


    #update noise
    if (not slow) and (not imgdiff) and (total_noise is not None):
        #global current_noise
        current_noise = (noise_update_all(total_noise,gridsize,gridsize,DM_trials,widthtrials,writeonly=True),current_noise[1] + 1)

    #update last frame
    if not slow and not imgdiff and append_frame:
        #global last_frame
        save_last_frame(last_frame,full=True)
        printlog("Writing to last_frame.npy",output_file=processfile)
    elif slow and not imgdiff and append_frame:
        save_last_frame(last_frame_slow,full=True,slow=True)
        printlog("Writing to last_frame_slow.npy",output_file=processfile)
    
    if cuda and (attach_fullimg_slow is not None) and attach['slow']['append_frame']:
        save_last_frame(last_frame_slow,full=True,slow=True)
        printlog("Writing to last_frame_slow.npy",output_file=processfile)
    searchlock.release()

    srchtime = time.time()-timing1
    srchtxtime = time.time()
    if savesearch or cands_found or fprtest:
        if (not fprtest):# and (not realtime):
            #save image
            f = open(cand_dir + "raw_cands/" + fullimg.img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (not slow and imgdiff) else "") + ".npy","wb")
            np.save(f,fullimg.image_tesseract_binned)
            f.close()

            #save fits
            numpy_to_fits(fullimg.image_tesseract_binned.astype(np.float32),cand_dir + "raw_cands/" + fullimg.img_id_isot  + ("_slow" if slow else "") +("_imgdiff" if (imgdiff and not slow) else "") +".fits")
            
            #save fits
            numpy_to_fits(fullimg.image_tesseract_searched.astype(np.float32),cand_dir + "raw_cands/" + fullimg.img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") +"_searched.fits")
            
        #save image OR if realtime, write to psrdada buffers
        """
        if False:#realtime:
            printlog(">>>SHAPES>>>"+str(fullimg.image_tesseract_searched.shape) + str(fullimg.image_tesseract.shape),output_file=processfile)
            if not (slow or imgdiff):
                rtwrite(fullimg.image_tesseract_searched,key=NSFRB_SRCHDADA_KEY)
                rtwrite(fullimg.image_tesseract,key=NSFRB_CANDDADA_KEY)
                rtwrite(TOAs,key=NSFRB_TOADADA_KEY)
            elif slow:
                rtwrite(fullimg.image_tesseract_searched,key=NSFRB_SRCHDADA_SLOW_KEY)
                rtwrite(fullimg.image_tesseract,key=NSFRB_CANDDADA_SLOW_KEY)
                rtwrite(TOAs,key=NSFRB_TOADADA_SLOW_KEY)
            #elif imgdiff:
            #    rtwrite(fullimg.image_tesseract_searched,key=NSFRB_SRCHDADA_SLOW_KEY)
            #    rtwrite(fullimg.image_tesseract,key=NSFRB_CANDDADA_SLOW_KEY)
            #    rtwrite(TOAs,key=NSFRB_TOADADA_SLOW_KEY)
        else:
        """
        f = open(cand_dir + "raw_cands/" + fullimg.img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_searched.npy","wb")
        np.save(f,fullimg.image_tesseract_searched)
        f.close()

        f = open(cand_dir + "raw_cands/" + fullimg.img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_TOAs.npy","wb")
        np.save(f,TOAs)
        f.close()
        
        f = open(cand_dir + "raw_cands/" + fullimg.img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + "_input.npy","wb")
        np.save(f,fullimg.image_tesseract)
        f.close()

        if fprtest:
            f = open(cand_dir + "fpr_test.csv","a")
            f.write("\n"+fullimg.img_id_isot + "," + str(np.nanmax(fullimg.image_tesseract_searched))) 
            f.close()
        elif fnrtest:
            f = open(cand_dir + "fnr_test.csv","a")
            f.write("\n"+fullimg.img_id_isot + "," + str(np.nanmax(fullimg.image_tesseract_searched)))
            f.close()

        #if the dask scheduler is set up, put the cand file name in the queue
        #if 'DASKPORT' in os.environ.keys() and QSETUP:
        #    QQUEUE.put("candidates_" + fullimg.img_id_isot + ".csv")

    if (cuda and (attach_fullimg_slow is not None)) and (savesearch or attach['slow']['cands_found'] or fprtest):
        if (not fprtest):# and (not realtime):
            #save image
            f = open(cand_dir + "raw_cands/" + attach_fullimg_slow.img_id_isot + "_slow.npy","wb")
            np.save(f,attach_fullimg_slow.image_tesseract_binned)
            f.close()

            #save fits
            numpy_to_fits(attach_fullimg_slow.image_tesseract_binned.astype(np.float32),cand_dir + "raw_cands/" + attach_fullimg_slow.img_id_isot  + "_slow.fits")
            
            #save fits
            numpy_to_fits(attach_fullimg_slow.image_tesseract_searched.astype(np.float32),cand_dir + "raw_cands/" + attach_fullimg_slow.img_id_isot + "_slow_searched.fits")

        f = open(cand_dir + "raw_cands/" + attach_fullimg_slow.img_id_isot + "_slow_searched.npy","wb")
        np.save(f,attach_fullimg_slow.image_tesseract_searched)
        f.close()

        f = open(cand_dir + "raw_cands/" + attach_fullimg_slow.img_id_isot + "_slow_TOAs.npy","wb")
        np.save(f,attach['slow']['TOAs'])
        f.close()

        f = open(cand_dir + "raw_cands/" + attach_fullimg_slow.img_id_isot + "_slow_input.npy","wb")
        np.save(f,attach_fullimg_slow.image_tesseract)
        f.close()

    if (cuda and (attach_fullimg_imgdiff is not None)) and (savesearch or attach['imgdiff']['cands_found'] or fprtest):
        if (not fprtest):# and (not realtime):
            #save image
            f = open(cand_dir + "raw_cands/" + attach_fullimg_imgdiff.img_id_isot + "_imgdiff.npy","wb")
            np.save(f,attach_fullimg_imgdiff.image_tesseract_binned)
            f.close()

            #save fits
            numpy_to_fits(attach_fullimg_imgdiff.image_tesseract_binned.astype(np.float32),cand_dir + "raw_cands/" + attach_fullimg_imgdiff.img_id_isot  + "_imgdiff.fits")
            
            #save fits
            numpy_to_fits(attach_fullimg_imgdiff.image_tesseract_searched.astype(np.float32),cand_dir + "raw_cands/" + attach_fullimg_imgdiff.img_id_isot + "_imgdiff_searched.fits")

        f = open(cand_dir + "raw_cands/" + attach_fullimg_imgdiff.img_id_isot + "_imgdiff_searched.npy","wb")
        np.save(f,attach_fullimg_imgdiff.image_tesseract_searched)
        f.close()

        f = open(cand_dir + "raw_cands/" + attach_fullimg_imgdiff.img_id_isot + "_imgdiff_TOAs.npy","wb")
        np.save(f,attach['imgdiff']['TOAs'])
        f.close()

        f = open(cand_dir + "raw_cands/" + attach_fullimg_imgdiff.img_id_isot + "_imgdiff_input.npy","wb")
        np.save(f,attach_fullimg_imgdiff.image_tesseract)
        f.close()

    srchtxtime = time.time()-srchtxtime
    fulltime=time.time()-timing1
    printlog(fullimg.image_tesseract_searched,output_file=processfile)
    printlog("done, total search time: " + str(np.around(fulltime,2)) + " s",output_file=processfile)
    ftime = open(timelogfile,"a")
    ftime.write("[search] " + str(fulltime)+"\n")
    ftime.close()

    ftime = open(srchtx_file,"a")
    ftime.write(str(srchtxtime) + "\n")
    ftime.close()

    ftime = open(srchtime_file,"a")
    ftime.write(str(srchtime) + "\n")
    ftime.close()

    if attach_mode:
        ret1 = []
        ret2 = []
        ret3 = []
        if cands_found:
            ret1.append("candidates_" + fullimg.img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + ".csv")
            ret2.append(slow)
            ret3.append(imgdiff)
        if 'slow' in attach.keys() and attach['slow']['cands_found']:
            ret1.append("candidates_" + attach_fullimg_slow.img_id_isot + "_slow.csv")
            ret2.append(True)
            ret3.append(False)
        if 'imgdiff' in attach.keys() and attach['imgdiff']['cands_found']:
            ret1.append("candidates_" + attach_fullimg_imgdiff.img_id_isot + "_imgdiff.csv")
            ret2.append(False)
            ret3.append(True)
        return ret1,ret2,ret3,srchtime,srchtxtime
    else:

        if cands_found:
            return "candidates_" + fullimg.img_id_isot + ("_slow" if slow else "") + ("_imgdiff" if (imgdiff and not slow) else "") + ".csv",slow,imgdiff,srchtime,srchtxtime
        else:
            return None,slow,imgdiff,srchtime,srchtxtime





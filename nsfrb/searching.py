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
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from pytorch_dedispersion import dedispersion,boxcar_filter,candidate_finder

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
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/") 
from nsfrb.config import *
from nsfrb.noise import noise_update,noise_dir,noise_update_all
from nsfrb import jax_funcs

#output_dir = cwd + "/tmpoutput/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/"
coordfile = cwd + "/DSA110_Station_Coordinates.csv" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/DSA110_Station_Coordinates.csv"
output_file = cwd + "-logfiles/search_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt"
cand_dir = cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
processfile = cwd + "-logfiles/process_log.txt"
frame_dir = cwd + "-frames/"
psf_dir = cwd + "-PSF/"
f=open(output_file,"w")
f.close()


try:
    torch.multiprocessing.set_start_method("spawn")
except:
    printlog("Failed to set torch multiprocess method...sucks for you",output_file=output_file)


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
"""
Search parameters
"""
#tsamp = 130 #ms
#T = 3250 #ms
#nsamps = int(T/tsamp)
#fmax  = 1530 #MHz
#fmin = 1280 #MHz
#c = 3e8 #m/s
#fc = 1400 #MHz
#lambdac = (c/(fc*1e6)) #m
#nchans = 16 #16 coarse channels
#chanbw = (fmax-fmin)/nchans #MHz
#telescope_diameter = 4.65 #m


#resolution parameters
#pixsize = 0.002962513099862611#(48/3600)*np.pi/180 #rad
#gridsize = 32#256
#RA_point = 0 #rad
#DEC_point = 0 #rad


#initialize jax device so we alternate between them if only 1 batch
jaxdev = 0


#create axes
RA_axis = np.linspace(RA_point-(pixsize*gridsize/2),RA_point+(pixsize*gridsize/2),gridsize)
DEC_axis = np.linspace(DEC_point-(pixsize*gridsize/2),DEC_point+(pixsize*gridsize/2),gridsize)
time_axis = np.linspace(0,T,nsamps) #ms
freq_axis = np.linspace(fmin,fmax,nchans) #MHz

#DM trials
def gen_dm(dm1,dm2,tol,nu,nchan,tsamp,B):
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

    #print('DM trials:',ndms)
    return dms

def gen_dm_shifts(DM_trials,freq_axis,tsamp,nsamps,gridsize=1,outputwraps=False): #note, you shouldn't need to set gridsize
    nDM = len(DM_trials)
    nchans = len(freq_axis)
    fmin =np.nanmin(freq_axis)
    fmax = np.nanmax(freq_axis)

    tdelays = -(((DM_trials[:,np.newaxis].repeat(nchans,axis=1))*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))).transpose())

    tdelaysall = np.zeros((nchans*2,nDM),dtype=np.int16) #jnp.device_put(jnp.zeros((len(DM_trials),nchans*2),dtype=jnp.int16),jax.devices()[0])
    tdelaysall[1::2,:] = (np.array(np.ceil(tdelays/tsamp),dtype=np.int8))
    tdelaysall[0::2,:] = (np.array(np.floor(tdelays/tsamp),dtype=np.int8))
    tdelays_frac = np.concatenate([tdelays/tsamp - tdelaysall[0::2,:],1 - (tdelays/tsamp - tdelaysall[0::2,:])],axis=0).transpose()

    #rearrange shift idxs and expand axes

    #--case 1: appending previous frame
    tDM_max = (4.15)*np.max(DM_trials)*((1/fmin/1e-3)**2 - (1/fmax/1e-3)**2) #ms
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


minDM = 171
maxDM = 4000
DM_trials = np.array(gen_dm(minDM,maxDM,1.5,fc*1e-3,nchans,tsamp,chanbw))#[0:1]
nDMtrials = len(DM_trials)

corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append = gen_dm_shifts(DM_trials,freq_axis,tsamp,nsamps)

#make boxcar filters in advance
widthtrials = np.array(2**np.arange(5),dtype=int)
nwidths = len(widthtrials)
def gen_boxcar_filter(widthtrials,truensamps,gridsize=1,nDMtrials=1): #note, you shouldn't need to set gridsize OR DM trials
    nwidths = len(widthtrials)
    boxcar = np.zeros((nwidths,gridsize,gridsize,truensamps,nDMtrials))
    loc = int(truensamps//2)
    for i in range(nwidths):
        wid = widthtrials[i]
        boxcar[i,:,:,loc-wid//2-2:loc+wid-wid//2-2,:] = 1

    return boxcar
full_boxcar_filter = gen_boxcar_filter(widthtrials,nsamps)

#get the current noise map from file
current_noise = noise_update_all(None,gridsize,gridsize,DM_trials,widthtrials,readonly=True) #noise.get_noise_dict(gridsize,gridsize)

#DM_trials = np.concatenate([[0],DM_trials])
#DM_trials = np.array([0,1000])#np.array([0,100])
#nDMtrials = len(DM_trials)

#snr threshold
SNRthresh = 6

#last image frame
tDM_max = (4.15)*np.max(DM_trials)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms
maxshift = int(np.ceil(tDM_max/tsamp))
def init_last_frame(gridsize_DEC,gridsize_RA,nsamps,nchans,frame_dir=frame_dir):
    noise = np.zeros((gridsize_DEC,gridsize_RA,nsamps,nchans))
    #check if raw noise file exists
    if len(glob.glob(noise_dir + "raw_noise_" + str(gridsize_DEC) + "x" + str(gridsize_RA) + ".npy")) > 0:
        raw_noise = np.load(noise_dir + "raw_noise_" + str(gridsize_DEC) + "x" + str(gridsize_RA) + ".npy")
        for i in range(nchans):
            noise[:,:,:,i] = norm.rvs(loc=0,scale=raw_noise[i],size=(gridsize_DEC,gridsize_RA,nsamps))
    f = open(frame_dir + "last_frame.npy","wb")
    np.save(f,noise)
    f.close()
    return

    
    

def save_last_frame(image_tesseract,full=False,maxDM=np.max(DM_trials),tsamp=tsamp,frame_dir=frame_dir):
    """
    This function writes the given frame to the npy file to store for dedispersion
    on the next timestep
    """

    #if full is set, save the full image; otherwise, only save the samples needed for
    #dedisperion to the maximum value
    if full:
        f = open(frame_dir + "last_frame.npy","wb")
        np.save(f,image_tesseract)
        f.close()
    else:
        tDM_max = (4.15)*maxDM*((1/fmin/1e-3)**2 - (1/fmax/1e-3)**2) #ms
        maxshift = int(np.ceil(tDM_max/tsamp))
        f = open(frame_dir + "last_frame.npy","wb")
        np.save(f,image_tesseract[:,:,-maxshift:,:])
        f.close()
def get_last_frame(frame_dir=frame_dir,maxDM=np.max(DM_trials)):
    f = open(frame_dir + "last_frame.npy","rb")
    image_tesseract = np.load(f)
    f.close()
    return image_tesseract

last_frame = get_last_frame()


#antenna positions
import csv
ANTENNALONS = []
ANTENNALATS = []
ANTENNAELEVS = []
with open(coordfile,'r') as csvfile:
    rdr = csv.reader(csvfile,delimiter=',')
    i = 0
    for row in rdr:
        #print(row)
        if row[1][:3] == 'DSA' and row[1] != 'DSA-110 Station Coordinates':
            ANTENNALATS.append(float(row[2]))
            ANTENNALONS.append(float(row[3]))
            if row[4] == '':
                ANTENNAELEVS.append(np.nan)
            else:
                ANTENNAELEVS.append(float(row[4]))
csvfile.close()

ANTENNALATS = np.array(ANTENNALATS)
ANTENNALONS = np.array(ANTENNALONS)
ANTENNAELEVS = np.array(ANTENNAELEVS)
#default_PSF = sim.make_PSF_cube()
#default_PSF_params = (gridsize,

"""
pre-computed psf values
"""
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
if gridsize in PSF_dict.keys():
    printlog("loading PSF for gridsize " + str(gridsize) +", declination " + str(PSF_dict[gridsize]['declabels'][np.argmin(np.abs(PSF_dict[gridsize]['declabels']-np.nanmean(DEC_axis)))]),output_file=processfile)
    default_PSF = np.array(np.load(psf_dir + "gridsize_" + str(gridsize) + "/PSF_" + str(gridsize) + "_{d:.2f}".format(d=PSF_dict[gridsize]['declabels'][np.argmin(np.abs(PSF_dict[gridsize]['declabels']-np.nanmean(DEC_axis)))]) + "_deg.npy"),dtype=np.float32)[:,:,np.newaxis,:].repeat(nsamps,axis=2)
    default_PSF_params = (gridsize,PSF_dict[gridsize]['declabels'][np.argmin(np.abs(PSF_dict[gridsize]['declabels']-np.nanmean(DEC_axis)))])
else:
    default_PSF = scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),gridsize//2,True,nsamps)#sim.make_PSF_cube()
    default_PSF_params = (gridsize,np.nanmean(DEC_axis))




"""
pre-computed cutoff pixels
"""
def get_RA_cutoff(dec,T=T,pixsize=pixsize):
    """
    dec: current declination
    T: integration time in milliseconds
    """
    cutoff_as = (T/1000)*15/np.cos(dec*np.pi/180) #arcseconds
    cutoff_pix = (cutoff_as/3600)//pixsize
    return int(cutoff_pix)
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
        """
        for j in range(nchans):
            PSFimg_reshaped = (PSF_kernel[:,:,0,j])[np.newaxis,np.newaxis,:,:]
            PSFimg_reshaped.to(device)

            for i in range(nsamps):        
                image_tesseract_reshaped = (image_tesseract[:,:,i,j])[np.newaxis,np.newaxis,:,:]
                image_tesseract_reshaped.to(device)
                
                #convolve
                image_tesseract_filtered[:,:,i,j] = tf.conv2d(image_tesseract_reshaped.cuda(),PSFimg_reshaped.cuda(),padding='same')[0,0,:,:]
                image_tesseract_reshaped.to("cpu")
                del image_tesseract_reshaped
            PSFimg_reshaped.to("cpu")
            del PSFimg_reshaped
        """
        """
            image_tesseract_reshaped = ((image_tesseract[:,:,:,j].transpose(0,2)).transpose(1,2))[:,np.newaxis,:,:]###.reshape((nsamps,1,gridsize_RA,gridsize_DEC))
            PSFimg_reshaped = (PSFimg[:,:,0,j])[np.newaxis,np.newaxis,:,:]#.reshape((1,1,gridsize_RA,gridsize_DEC))
            image_tesseract_reshaped.to(device)
            PSFimg_reshaped.to(device)
            #convolve
            image_tesseract_filtered[:,:,:,j] = (tf.conv2d(image_tesseract_reshaped,PSFimg_reshaped,padding='same')[:,0,:,:].transpose(1,2)).transpose(0,2)#.reshape((gridsize_RA,gridsize_DEC,nsamps))
        """
        #image_tesseract_reshaped.to("cpu")
        #PSFimg_reshaped.to("cpu")
        #del image_tesseract_reshaped
        #del PSFimg_reshaped
        #torch.cuda.empty_cache()


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
            np.save(f,noisemap.numpy())
            f.close()
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
        """
        
        if samenoise:
            csig_filtered = csig_all[0,0,~torch.isnan(csig_all[0,0,:]).bool()]
            s=torch.std(csig_filtered[csig_filtered<noiseth*torch.max(csig_filtered)])
            noisemap[:,:] = s
        for i in range(gridsize_DEC):
            for j in range(gridsize_RA):
                csig_filtered = csig_all[i,j,~torch.isnan(csig_all[i,j,:]).bool()]
                mn = torch.median(csig_filtered[csig_filtered<noiseth*torch.max(csig_filtered)])
                image_tesseract_binned[i,j] = torch.max(csig_filtered - mn)
                if not samenoise:
                    noisemap[i,j] = torch.std(csig_filtered[csig_filtered<noiseth*torch.max(csig_filtered)])
         
                if plot:
                    csig = csig_all[i,j,:]
                    plt.subplot(1,4,3)
                    plt.plot(csig,color='grey',alpha=1)
                    plt.plot(np.arange(len(csig))[csig<noiseth],csig[csig<noiseth],color='red',marker='o',linestyle='')
                    plt.axvline(noiseth/wid)
                    #plt.axvline(np.argmin(csig-np.max(csig)/2),color='red')
                    #plt.axvline(nsamps-np.argmin(csig[::-1]-np.max(csig)/2),color='purple')

                    plt.subplot(1,4,1)
                    plt.plot(timeseries,color='grey',alpha=1)
        print("noisemap:",noisemap,sum(torch.isnan(noisemap)),file=fout)
        print("img:",image_tesseract_binned,sum(torch.isnan(image_tesseract_binned)),file=fout)
        image_tesseract_binned = image_tesseract_binned/noisemap
        """
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
        """for i in range(gridsize_DEC):
            for j in range(gridsize_RA):
                timeseries = image_tesseract_filtered_dm[i,j,:]
                csig = np.convolve(np.nan_to_num(timeseries,nan=0),boxcar,'same')#/wid#/np.sum(boxcar)
                peakidx = np.argmax(csig)


            
                #off-pulse mean and standard deviation 
                if not samenoise or (i == 0 and j == 0):
                    s=np.nanstd(csig[csig<noiseth*np.nanmax(csig)])#np.nanstd(np.concatenate([csig[:np.nanargmax(csig)-wid],csig[np.nanargmax(csig)+wid+1:]]))
                mn=np.nanmedian(csig[csig<noiseth*np.nanmax(csig)])
                noisemap[i,j] = s

            

                #print(np.nanmax(csig),s,np.nanmax(csig)/s)
                if s == 0: image_tesseract_binned[i,j] = np.nan
                else: image_tesseract_binned[i,j] = np.nanmax(csig - mn)/s#/wid
                #print(np.nanargmax(csig))
                if plot:
                    plt.subplot(1,4,3)
                    plt.plot(csig,color='grey',alpha=1)
                    plt.plot(np.arange(len(csig))[csig<noiseth],csig[csig<noiseth],color='red',marker='o',linestyle='')
                    plt.axvline(noiseth/wid)
                    #plt.axvline(np.argmin(csig-np.max(csig)/2),color='red')
                    #plt.axvline(nsamps-np.argmin(csig[::-1]-np.max(csig)/2),color='purple')
                
                    plt.subplot(1,4,1)
                    plt.plot(timeseries,color='grey',alpha=1)
        """
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


def dedisp_pad_with(vector,pad_width,iaxis,kwargs):
    #we want to make a padding function which allows different number of pad values for each row
    padvals= kwargs.get('padvals',0) #padvals is a tuple where each element is an array with length equal to the number of rows. This stores the amount to pad each row with
    endflag = kwargs.get('endflag',False) #False == pad start, True == pad end
    device = kwargs.get('device',None)
    cuda = device.type == 'cuda'
    currpad_dict = kwargs.get('currpad_dict',dict())
    currpad = currpad_dict['currpad']
    ax = kwargs.get("ax",0)
    dd_value = kwargs.get("dd_value",0)



    if iaxis != ax: return
    #we will cut the array to be the same size as the input, so pad_width == 0
    assert(pad_width == (0,0))
    

    #use the currpad value to index each padvalue
    if endflag:
        padshape = (0,padvals[currpad])
    else:
        padshape = (padvals[currpad],0)
    if not cuda:
        if endflag:
            vector[:] = np.pad(vector,padshape,mode='constant',constant_values=dd_value)[-len(vector):]
        else:
            vector[:] = np.pad(vector,padshape,mode='constant',constant_values=dd_value)[:len(vector)]
    else:
        vector_torch = torch.from_numpy(vector).to(device)
        if endflag:
            vector[:] = tf.pad(vector_torch,padshape,mode='constant',value=dd_value).to("cpu").numpy()[-len(vector):]
        else:
            vector[:] = tf.pad(vector_torch,padshape,mode='constant',value=dd_value).to("cpu").numpy()[:len(vector)]
        vector_torch.to("cpu")
        del vector_torch
        torch.cuda.empty_cache()

    currpad_dict['currpad'] += 1
    if currpad_dict['currpad'] >= len(padvals): currpad_dict['currpad'] = 0
    return 

def dedisp_pad(x,tdelays_idx,device,endflag):
    currpad_dict = {"currpad":0}
    return np.pad(x,(0,0),dedisp_pad_with,padvals=tdelays_idx,device=device,currpad_dict=currpad_dict,ax=2,dd_value=0,endflag=endflag)



# Brute force dedispersion
def dedisperse(image_tesseract_point,DM,tsamp=tsamp,freq_axis=freq_axis,device=None,output_file="",append_last_frame=True):
    """
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
        """#shift each channel
        image_tesseract_point.to(device)
        if neg_flag:
            arrlow = torch.from_numpy(dedisp_pad(image_tesseract_point,tdelays_idx_low,device=device,endflag=False)).to(device)
            arrhi = torch.from_numpy(dedisp_pad(image_tesseract_point,tdelays_idx_hi,device=device,endflag=False)).to(device)
        else:
            arrlow = torch.from_numpy(dedisp_pad(image_tesseract_point,tdelays_idx_low,device=device,endflag=True)).to(device)
            arrhi = torch.from_numpy(dedisp_pad(image_tesseract_point,tdelays_idx_hi,device=device,endflag=True)).to(device)
        dedisp_timeseries_all = (arrlow*(1-tdelays_frac) + arrhi*(tdelays_frac)).mean(3)
        dedisp_img = arrlow*(1-tdelays_frac) + arrhi*(tdelays_frac)
        image_tesseract_point.to("cpu")
        arrlow.to("cpu")
        arrhi.to("cpu")
        del arrlow
        del arrhi
        torch.cuda.empty_cache()


        """
        """
        #shift each channel
        for k in range(nchans):
            #move channel to GPU
            channel = image_tesseract_point[:,:,:,k]
            channel.to(device)
            
            
            #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac);
            if neg_flag:
                #padshape = (0,0)*(len(dedisp_timeseries_all.shape)-1) #+ [tdelays_idx_low[k],0] #tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-1) + [(tdelays_idx_low[k],0)])
                padshape = (tdelays_idx_low[k],0)
                arrlow =  tf.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",value=0)[:,:,:nsamps]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
            else:
                #padshape = (0,0)*(len(dedisp_timeseries_all.shape)-1) #+ [0,tdelays_idx_low[k]] #tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-1) + [(0,tdelays_idx_low[k])])
                padshape = (0,tdelays_idx_low[k])
                arrlow =  tf.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",value=0)[:,:,tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
            #print(padshape,file=fout)

            if neg_flag:
                #padshape = (0,0)*(len(dedisp_timeseries_all.shape)-1) #+ [tdelays_idx_hi[k],0] #tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-1) + [(tdelays_idx_hi[k],0)])
                padshape = (tdelays_idx_hi[k],0)
                arrhi =  tf.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",value=0)[:,:,:nsamps]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
            else:
                #padshape = (0,0)*(len(dedisp_timeseries_all.shape)-1) #+ [0,tdelays_idx_hi[k]] #tuple([(0,0)]*(len(dedisp_timeseries_all.shape)-1) + [(0,tdelays_idx_hi[k])])
                padshape = (0,tdelays_idx_hi[k])
                arrhi =  tf.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",value=0)[:,:,tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                #print(tf.pad(image_tesseract_point[:,:,:,k],padshape,mode="constant",value=0).shape,arrhi.shape,image_tesseract_point[:,:,:,k].shape)
                #print(padshape)

            #move from GPU
            #channel.to("cpu")

            print(padshape,file=fout)

            dedisp_timeseries_all += arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])
            dedisp_img[:,:,:,k] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])
        #move from GPU
        channel.to("cpu")
        del channel
        torch.cuda.empty_cache()
        """
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
        """if neg_flag:
            arrlow = dedisp_pad(image_tesseract_point,tdelays_idx_low,cuda=False,endflag=False)
            arrhi = dedisp_pad(image_tesseract_point,tdelays_idx_hi,cuda=False,endflag=False)
        else:
            arrlow = dedisp_pad(image_tesseract_point,tdelays_idx_low,cuda=False,endflag=True)
            arrhi = dedisp_pad(image_tesseract_point,tdelays_idx_hi,cuda=False,endflag=True)
        dedisp_timeseries_all = (arrlow*(1-tdelays_frac) + arrhi*(tdelays_frac)).mean(3)
        dedisp_img = arrlow*(1-tdelays_frac) + arrhi*(tdelays_frac)
        
        """
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



#Alternate search code using PyTorchDedispersion (Kosogorov in prep) from LWA
def run_PyTorchDedisp_search(image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,freq_axis=freq_axis,PSF=default_PSF,output_file="",
        DM_trials=DM_trials,widthtrials=widthtrials,tsamp=tsamp,SNRthresh=SNRthresh,verbose=False,space_filter=True,canddict=dict(),usefft=False):
    """
    This is a wrapper around the PyTorchDedispersion code defined in https://github.com/nkosogor/PyTorchDedispersion/, developed by
    Nikita Kosogorov for use on the LWA.
    """

    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    printprefix = ""#"[" + str(raidx_offset) + str(decidx_offset) + str(dm_offset) + "]"

    #check if cuda is available; if not, exit
    device = torch.device(random.choice(np.arange(torch.cuda.device_count(),dtype=int)) if torch.cuda.is_available() else "cpu")
    usingGPU = device.type == "cuda"
    if not usingGPU:
        print("GPUs not available, using CPUs",file=fout)
        return -1

    #get axis sizes
    gridsize_RA = len(RA_axis)
    gridsize_DEC = len(DEC_axis)
    gridsize = gridsize_RA
    nsamps = len(time_axis)
    nchans = len(freq_axis)
    nwidthtrials = len(widthtrials)
    nDMtrials = len(DM_trials)

    if space_filter:
        assert(gridsize_RA == gridsize_DEC)
        #create PSF if the shape doesn't match
        if PSF.shape != image_tesseract.shape:
            print(printprefix + "Updating PSF...",file=fout)
            PSF = sim.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,nchans=nchans)
        #2D matched filter for each timestep and channel
        print(printprefix +"Spatial matched filtering with DSA PSF...",file=fout)
        if usefft:
            print(printprefix +"Using 2D FFT method...",file=fout)

        if usingGPU:
            image_tesseract_filtered = matched_filter_space(torch.from_numpy(image_tesseract),torch.from_numpy(np.array(PSF,np.float16)),usefft=usefft,device=device,output_file=output_file).numpy()

        else:
            image_tesseract_filtered = matched_filter_space(image_tesseract,PSF,usefft=usefft,device=device)
        print(printprefix +"Done!",file=fout)
        print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_filtered))),file=fout)
    else:
        image_tesseract_filtered = image_tesseract


    #loop over each pixel
    image_tesseract_binned = np.zeros((gridsize_DEC,gridsize_RA,nwidthtrials,nDMtrials))

    cands = []
    candidxs = []
    candra_idxs = []
    canddec_idxs = []
    candwid_idxs = []
    canddm_idxs = []
    candsnrs = []
    for i in range(gridsize_RA):
        for j in range(gridsize_DEC):
            #get data tensor
            data_tensor = torch.from_numpy(image_tesseract_filtered[j,i,:,:].transpose()).to(device)

            #get freq tensor
            frequencies_tensor = torch.from_numpy(freq_axis).to(device)
            freq_start = torch.min(frequencies_tensor)

            #get dm range tensor
            dm_range = torch.from_numpy(DM_trials).to(device)

            #get time res tensor
            timeax_tensor = torch.from_numpy(time_axis)
            time_resolution = time_axis[1]-time_axis[0]

            
            #get width tensor
            widths = torch.from_numpy(widthtrials).to(device)

            #window size and snr threshold
            window_size=len(time_axis)//2
            snr_threshold = SNRthresh
    
            #create dedisp, boxcar, threshold plans
            dedisp_obj = dedispersion.Dedispersion(data_tensor,
                                                    frequencies_tensor,
                                                    dm_range,
                                                    freq_start,
                                                    time_resolution)
            dedisp_data = (dedisp_obj.perform_dedispersion()).sum(dim=2)

            boxcar_obj = boxcar_filter.BoxcarFilter(dedisp_data)
            filt_data = boxcar_obj.apply_boxcar(widths)

            cand_obj = candidate_finder.CandidateFinder(filt_data,window_size)
            cand_data = cand_obj.find_candidates(snr_threshold,widths,remove_trend=True)
            snr_data = cand_obj.calculate_snr(filt_data)

            #add to full result arrays
            image_tesseract_binned[j,i,:,:] = snr_data.to("cpu").numpy().max(2)
            cands = np.concatenate([cands,[(RA_axis[i],DEC_axis[j],cand_data[k]['Boxcar Width'],DM_trials[cand_data[k]['DM Index']],cand_data[k]['SNR']) for k in range(len(cand_data))]])
            candidxs = np.concatenate([candidxs,[(i,j,list(widthtrials).index(cand_data[k]['Boxcar Width']),cand_data[k]['DM Index'],cand_data[k]['SNR']) for k in range(len(cand_data))]])

            candra_idxs = np.concatenate([candra_idxs,[i]*len(cand_data)])
            canddec_idxs = np.concatenate([canddec_idxs,[j]*len(cand_data)])
            candwid_idxs = np.concatenate([candwid_idxs,[list(widthtrials).index(cand_data[k]['Boxcar Width']) for k in range(len(cand_data))]])
            canddm_idxs = np.concatenate([canddm_idxs,[cand_data[k]['DM Index'] for k in range(len(cand_data))]])
            candsnrs = np.concatenate([candsnrs,[cand_data[k]['SNR'] for k in range(len(cand_data))]])


    data_tensor.to("cpu")
    frequencies_tensor.to("cpu")
    dm_range.to("cpu")
    widths.to("cpu")
    del data_tensor
    del frequencies_tensor
    del dm_range
    del widths
    torch.cuda.empty_cache()
        

    #make a dictionary for easy plotting of results
    canddict['ra_idxs'] = copy.deepcopy(candra_idxs)
    canddict['dec_idxs'] = copy.deepcopy(canddec_idxs)
    canddict['wid_idxs'] = copy.deepcopy(candwid_idxs)
    canddict['dm_idxs'] = copy.deepcopy(canddm_idxs)        
    canddict['ras'] = RA_axis[np.array(candra_idxs,dtype=int)]
    canddict['decs'] = DEC_axis[np.array(canddec_idxs,dtype=int)]
    canddict['wids'] = widthtrials[np.array(candwid_idxs,dtype=int)]
    canddict['dms'] = DM_trials[np.array(canddm_idxs,dtype=int)]        
    canddict['snrs'] = copy.deepcopy(candsnrs)
    print(printprefix +"Done! Found " + str(len(candsnrs)) + " candidates",file=fout)
    if output_file != "":
        fout.close()
    return candidxs,cands,image_tesseract_binned,image_tesseract_filtered,canddict,DM_trials
    
#Updated search code
#run search pipeline with desired DM, width trial range; output candidates to a csv? pkl? txt?
#takes 4D cube (RA,DEC,TIME,FREQUENCY)

def run_search_new(image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,freq_axis=freq_axis,
                   DM_trials=DM_trials,widthtrials=widthtrials,tsamp=tsamp,SNRthresh=SNRthresh,plot=False,
                   off=10,PSF=default_PSF,offpnoise=0.3,verbose=False,output_file="",noiseth=0.9,canddict=dict(),usefft=False,
                   multithreading=False,nrows=1,ncols=1,space_filter=True,raidx_offset=0,decidx_offset=0,dm_offset=0,
                   threadDM=False,samenoise=False,cuda=False,exportmaps=False,kernel_size=len(RA_axis),append_frame=True,DMbatches=1,SNRbatches=1,usejax=True,RA_cutoff=default_cutoff):

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


    if cuda: 
        multithreading = False
        print("CUDA flag overrides multithreading",file=fout)
    #find available cuda GPU to use
    if cuda:
        device = torch.device(random.choice(np.arange(torch.cuda.device_count(),dtype=int)) if torch.cuda.is_available() else "cpu")
        usingGPU = device.type == "cuda"
        if not usingGPU:
            print("GPUs not available, using CPUs",file=fout)
    else:
        device = None
        usingGPU = False
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
        assert(gridsize_RA == gridsize_DEC)

        #get PSF if possible
        #if gridsize in PSF_dict.keys():
        #    PSF = PSF_dict[gridsize][PSF_decs[np.argmin(PSF_dec_bins-np.mean(DEC_axis))]]

        #create PSF if the shape doesn't match
        #if PSF.shape != image_tesseract.shape:
        #    print(printprefix + "Updating PSF...",file=fout)
        #    PSF = sim.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,nchans=nchans)
        #2D matched filter for each timestep and channel
        print(printprefix +"Spatial matched filtering with DSA PSF...",file=fout)
        if usefft:
            print(printprefix +"Using 2D FFT method...",file=fout)
        

        if usefft and usingGPU and usejax and DMbatches > 1:
            #make empty array
            image_tesseract_filtered = jax_funcs.matched_filter_fft_jit(jax.device_put(np.array(image_tesseract,dtype=np.float32),jax.devices()[0]),jax.device_put(np.array(PSF,dtype=np.float32),jax.devices()[0]))
            """
            np.zeros(image_tesseract.shape,dtype=image_tesseract.dtype)
    
            executor = ThreadPoolExecutor(DMbatches*DMbatches)
            task_list = []
            subband_size = int(nchans//(DMbatches))#*DMbatches))
            for i in range(DMbatches):#*DMbatches):
                
                if i%2 == 0:
                    task_list.append(executor.submit(jax_funcs.matched_filter_fft_jit,jax.device_put(np.array(image_tesseract[:,:,:,i*subband_size:(i+1)*subband_size],dtype=np.float32),jax.devices()[0]),jax.device_put(np.array(PSF[:,:,:,i*subband_size:(i+1)*subband_size],dtype=np.float32),jax.devices()[0]),i))
                else:
                    task_list.append(executor.submit(jax_funcs.matched_filter_fft_jit,jax.device_put(np.array(image_tesseract[:,:,:,i*subband_size:(i+1)*subband_size],dtype=np.float32),jax.devices()[1]),jax.device_put(np.array(PSF[:,:,:,i*subband_size:(i+1)*subband_size],dtype=np.float32),jax.devices()[1]),i))
            #get results
            for t in task_list:
                outtup = t.result()
                i = outtup[1]
                image_tesseract_filtered[:,:,:,i*subband_size:(i+1)*subband_size] =np.array(outtup[0])
            executor.shutdown()
            """
            print(printprefix +"Done!",file=fout)
            print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_filtered))),file=fout)
            print("Time for Space Filter: " + str(time.time()-t1) + " s",file=fout)
        elif usefft and usingGPU and usejax and DMbatches == 1:
            print("Using combined jit with 1 batch",file=fout)
            image_tesseract_filtered = image_tesseract
        elif usingGPU:
            image_tesseract_filtered = matched_filter_space(torch.from_numpy(image_tesseract),torch.from_numpy(PSF),kernel_size=kernel_size,usefft=usefft,device=device,output_file=output_file).numpy()
            print(printprefix +"Done!",file=fout)
            print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_filtered))),file=fout)
            print("Time for Space Filter: " + str(time.time()-t1) + " s",file=fout)
        else:
            image_tesseract_filtered = matched_filter_space(image_tesseract,PSF,kernel_size=kernel_size,usefft=usefft,device=device)
    else: 
        image_tesseract_filtered = image_tesseract
    #print("POST-FILTER SHAPE: " + str(image_tesseract_filtered.shape) + ","  + str(PSF.shape),file=fout)
    #print(gridsize,file=fout)
    

    #REVISED IMPLEMENTATION: DO DEDISP AND BOXCAR TOGETHER SO WE DON'T HAVE TO KEEP MOVING TO/FROM GPU
    total_noise=None
    if usefft and usingGPU and usejax:
        t1 = time.time()
        nwidths,ndms = len(widthtrials),len(DM_trials)
        #get data from previous timeframe
        if append_frame:
            global last_frame

            truensamps = image_tesseract_filtered.shape[2]
            image_tesseract_filtered_cut = np.concatenate([last_frame[:,RA_cutoff:,:,:],image_tesseract_filtered[:,:-RA_cutoff,:,:]],axis=2)
            PSF = PSF[:,int(RA_cutoff//2):-(RA_cutoff - int(RA_cutoff//2)),:,:]
            nsamps = image_tesseract_filtered.shape[2]
            corr_shifts_all = corr_shifts_all_append
            tdelays_frac = tdelays_frac_append
            
            print("Appending data from previous timeframe, new shape: " + str(image_tesseract_filtered_cut.shape),file=fout)
        
            #save frame
            last_frame = image_tesseract_filtered[:,:,-maxshift:,:]
            #save_last_frame(image_tesseract_filtered)
            #print("Writing to last_frame.npy",file=fout)
            RA_axis = RA_axis[:-RA_cutoff]
            gridsize_RA = len(RA_axis)
        else:
            corr_shifts_all = corr_shifts_all_no_append
            tdelays_frac = tdelays_frac_no_append
            truensamps = nsamps = image_tesseract_filtered.shape[2]
            image_tesseract_filtered_cut=image_tesseract_filtered_cut
        #subgrid
        subgridsize_RA = gridsize_RA//DMbatches
        subgridsize_DEC = gridsize_DEC//DMbatches
        #boxcar = full_boxcar_filter
        
        #noise prep
        total_noise = torch.zeros((nwidths,ndms))
        
        prev_noise,prev_noise_N = current_noise #noise_update_all(None,gridsize_RA,gridsize_DEC,DM_trials,widthtrials,readonly=True)

        print("Time for Prep" + str(time.time()-t1),file=fout)
        t1 = time.time()

        if DMbatches > 1:
            executor = ThreadPoolExecutor(DMbatches*DMbatches)#maxProcesses)
            task_list = []
            image_tesseract_binned = np.zeros((gridsize_DEC,gridsize_RA,len(widthtrials),len(DM_trials)))
            for i in range(DMbatches):
                for j in range(DMbatches):
                    if j%2 == 0:
                        task_list.append(executor.submit(jax_funcs.dedisp_snr_fft_jit_0,jax.device_put(np.array(image_tesseract_filtered_cut[j*subgridsize_DEC:(j+1)*subgridsize_DEC,i*subgridsize_RA:(i+1)*subgridsize_RA,:,:],dtype=np.float32),jax.devices()[0]),jax.device_put(corr_shifts_all,jax.devices()[0]),jax.device_put(tdelays_frac,jax.devices()[0]),jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[0]),jax.device_put(np.array(prev_noise,dtype=np.float16),jax.devices()[0]),prev_noise_N,noiseth,i,j))
                    else:
                        task_list.append(executor.submit(jax_funcs.dedisp_snr_fft_jit_0,jax.device_put(np.array(image_tesseract_filtered_cut[j*subgridsize_DEC:(j+1)*subgridsize_DEC,i*subgridsize_RA:(i+1)*subgridsize_RA,:,:],dtype=np.float32),jax.devices()[1]),jax.device_put(corr_shifts_all,jax.devices()[1]),jax.device_put(tdelays_frac,jax.devices()[1]),jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[1]),jax.device_put(np.array(prev_noise,dtype=np.float16),jax.devices()[1]),prev_noise_N,noiseth,i,j))
                    


            #get results
            for t in task_list:
                outtup = t.result()
                i,j = outtup[2],outtup[3]
                image_tesseract_binned[j*subgridsize_DEC:(j+1)*subgridsize_DEC,i*subgridsize_RA:(i+1)*subgridsize_RA,:,:] =  np.array(outtup[0])
                total_noise += np.array(outtup[1])/(DMbatches**2)
            executor.shutdown()
        else:
            #jaxdev = random.choice(np.arange(len(jax.devices()),dtype=int))
            global jaxdev
            outtup = jax_funcs.matched_filter_dedisp_snr_fft_jit(jax.device_put(np.array(image_tesseract_filtered_cut,dtype=np.float32),jax.devices()[jaxdev]),
                                                                 jax.device_put(np.array(PSF[:,:,0:1,:],dtype=np.float32),jax.devices()[jaxdev]),
                                                                 jax.device_put(corr_shifts_all,jax.devices()[jaxdev]),
                                                                 jax.device_put(tdelays_frac,jax.devices()[jaxdev]),
                                                                 jax.device_put(np.array(full_boxcar_filter,dtype=np.float16),jax.devices()[jaxdev]),
                                                                 jax.device_put(np.array(prev_noise,dtype=np.float16),jax.devices()[jaxdev]),
                                                                 prev_noise_N,noiseth)
            image_tesseract_binned,total_noise = np.array(outtup[0]),np.array(outtup[1])
            
            jaxdev += 1 
            jaxdev %= 2 
        
        #noise_update_all(total_noise,gridsize_RA,gridsize_DEC,DM_trials,widthtrials,writeonly=True) 
        print("Time for DM and SNR:" + str(time.time()-t1),file=fout)

        #sort candidates
        t1 = time.time()
        print(printprefix +"Searching for candidates with S/N > " + str(SNRthresh) + "...",file=fout)
        #find candidates above SNR threshold
        #condition = (image_tesseract_binned>=SNRthresh).flatten()
        #ncands = np.sum(condition)
        #canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs=np.unravel_index(np.arange(gridsize_DEC*gridsize_RA*ndms*nwidths)[condition],(gridsize_DEC,gridsize_RA,nwidths,ndms))#[1].shape

        print(image_tesseract_binned.shape,image_tesseract_filtered_cut.shape,file=fout)
        canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs = np.nonzero(image_tesseract_binned>=SNRthresh)
        ncands = len(canddec_idxs)
        #print(len(DEC_axis),np.max(canddec_idxs),len(RA_axis),np.max(candra_idxs),file=fout)
        #fout.close()
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



    #use the concurrent futures package to search sub-images separately; we have to do this AFTER the spatial matched filter so that the PSF structure is suppressed
    elif multithreading: 
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


            #save the binned image and candidates
            candidxs = list(candidxs) + list(candidxs_i)
            cands = list(cands) + list(cands_i)
            if threadDM: image_tesseract_binned[decidx_offset_i:decidx_offset_i + gridsize_DEC_i,raidx_offset_i:raidx_offset_i + gridsize_RA_i,:,dm_offset_i:dm_offset_i+1] = image_tesseract_binned_i    
            else: image_tesseract_binned[decidx_offset_i:decidx_offset_i + gridsize_DEC_i,raidx_offset_i:raidx_offset_i + gridsize_RA_i,:,:] = image_tesseract_binned_i

            for k in canddict_i.keys():
                canddict[k] = np.concatenate([canddict[k],canddict_i[k]])

        #make a dictionary for easy plotting of results
        ncands = len(cands)
    else: #proceed normally

        t1 = time.time()
        #dedisperse --> gridsize x gridsize x time x DM
        nDMtrials = len(DM_trials)
        print(printprefix +"Starting dedispersion with " + str(nDMtrials) + " trials...",file=fout)
        #image_tesseract_dedisp = np.zeros((gridsize_DEC,gridsize_RA,nsamps,nDMtrials)) 
    

        if usingGPU:
            """
            for i in range(0,ndmbatches,2):
                miniExecutor = ProcessPoolExecutor(5)
                minitasklist = []
                minitasklist.append(miniExecutor.submit(dedisperse_allDM,torch.from_numpy(image_tesseract_filtered),DM_trials[int(i*dmbatchsize):int((i+1)*dmbatchsize)],tsamp,freq_axis,torch.device(0 if torch.cuda.is_available() else "cpu"),output_file,True,i))
                
                minitasklist.append(miniExecutor.submit(dedisperse_allDM,torch.from_numpy(image_tesseract_filtered),DM_trials[int((i+1)*dmbatchsize):int((i+2)*dmbatchsize)],tsamp,freq_axis,torch.device(1 if torch.cuda.is_available() else "cpu"),output_file,True,i+1))
               
                for minifuture in as_completed(minitasklist):
                    miniresult = minifuture.result()
                    j = miniresult[-1]
                    image_tesseract_dedisp[:,:,:,int(j*dmbatchsize):int((j+1)*dmbatchsize)] = miniresult[0]
                
            """
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
        """
        image_tesseract_dedisp = np.zeros((gridsize_DEC,gridsize_RA,nsamps,nDMtrials)) #stores output array as dedispersion transform for every pixel
        for d in range(nDMtrials):
            if usingGPU:
                image_tesseract_dedisp[:,:,:,d] = dedisperse(torch.from_numpy(image_tesseract_filtered),DM=DM_trials[d],tsamp=tsamp,freq_axis=freq_axis,output_file=output_file,device=device)[0].numpy()
            else:
                image_[:truensamps]tesseract_dedisp[:,:,:,d] = dedisperse(image_tesseract_filtered,DM=DM_trials[d],tsamp=tsamp,freq_axis=freq_axis,output_file=output_file,device=device)[0]
        """
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
        """
        #PSF parameters
        maxs = []
        maxs2 = []
        for w in range(nwidthtrials):
            for d in range(nDMtrials):
                if usingGPU:
                    image_tesseract_binned[:,:,w,d] = snr_vs_RA_DEC_new(torch.from_numpy(image_tesseract_dedisp[:,:,:,d]),widthtrials[w],DM_trials[d],noiseth=noiseth,output_file=output_file,samenoise=samenoise,device=device,exportmaps=exportmaps).numpy()
                else:
                    image_tesseract_binned[:,:,w,d] = snr_vs_RA_DEC_new(image_tesseract_dedisp[:,:,:,d],widthtrials[w],DM_trials[d],noiseth=noiseth,output_file=output_file,samenoise=samenoise,device=device,exportmaps=exportmaps) 
                if d ==0 and plot:
                    maxs.append(image_tesseract_binned[15, 16,w,d])
                elif plot:
                    maxs2.append(image_tesseract_binned[15, 16,w,d])
        """
        print(printprefix +"Done!",file=fout)    
        print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_binned))),file=fout)
        print("Time for boxcar filter: " + str(time.time()-t1) + " s",file=fout)
        """if plot:
            plt.figure(figsize=(12,6))
            plt.plot(widthtrials,maxs,'o-')
            #plt.plot(widthtrials,maxs2,'o-')
            #plt.plot(np.arange(1,10),maxs[np.argmin(np.abs(widthtrials-5))]*np.sqrt(5/np.arange(1,10)),color='red')
            #plt.plot(np.arange(1,10),maxs[np.argmin(np.abs(widthtrials-5))]*np.sqrt(np.arange(1,10)/5),color='blue')
            plt.show()
        """


        t1 = time.time()
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
    print(printprefix +"Done! Found " + str(ncands) + " candidates",file=fout)
    if output_file != "":
        fout.close()
    if append_frame:
        return candidxs,cands,image_tesseract_binned,image_tesseract_filtered[:,:,-truensamps:,:],canddict,DM_trials,raidx_offset,decidx_offset,dm_offset,total_noise
    else:
        return candidxs,cands,image_tesseract_binned,image_tesseract_filtered,canddict,DM_trials,raidx_offset,decidx_offset,dm_offset,total_noise





#CONTEXTSETUP = False
def search_task(fullimg,SNRthresh,subimgpix,model_weights,verbose,usefft,cluster,multithreading,nrows,ncols,threadDM,samenoise,cuda,toslack,PyTorchDedispersion,space_filter,kernel_size,exportmaps,savesearch,append_frame,DMbatches,SNRbatches,usejax,noiseth,nocutoff):
    #global CONTEXTSETUP
    #if not QSETUP and not CONTEXTSETUP:
    #    CONTEXTSETUP = True
    #    torch.multiprocessing.set_start_method("spawn")
    printlog("starting search process " + str(fullimg.img_id_isot) + "...",output_file=processfile,end='')


    #need to account for shift between successive images of 3.25 seconds
    """
    global default_cutoff
    if nocutoff:
        default_cutoff = 0
    else:
        default_cutoff = get_RA_cutoff(fullimg.DEC_axis[len(fullimg.DEC_axis)//2])
    """


    #define search params
    gridsize=fullimg.image_tesseract.shape[0]
    RA_axis = fullimg.RA_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
    DEC_axis= fullimg.DEC_axis#np.linspace(-gridsize//2,gridsize//2,gridsize)
    nsamps = fullimg.image_tesseract.shape[2]
    nchans = fullimg.image_tesseract.shape[3]
    time_axis = np.arange(nsamps)*tsamp
    global default_PSF
    global default_PSF_params
    if kernel_size in PSF_dict.keys() and default_PSF_params != (kernel_size,PSF_dict[kernel_size]['declabels'][np.argmin(np.abs(PSF_dict[kernel_size]['declabels']-np.nanmean(DEC_axis)))]):
        best_dec = PSF_dict[kernel_size]['declabels'][np.argmin(np.abs(PSF_dict[kernel_size]['declabels']-np.nanmean(DEC_axis)))]
        printlog("loading PSF for kernelsize " + str(kernel_size) +", declination " + str(best_dec),output_file=processfile)
        default_PSF = np.array(np.load(psf_dir + "gridsize_" + str(kernel_size) + "/PSF_" + str(kernel_size) + "_{d:.2f}".format(d=best_dec) + "_deg.npy"),dtype=np.float32)[:,:,np.newaxis,:].repeat(nsamps,axis=2)
        default_PSF_params = (kernel_size,best_dec)
    elif kernel_size not in PSF_dict.keys() and (default_PSF_params[0] != kernel_size or np.abs(default_PSF_params[1] - float("{d:.2f}".format(d=np.nanmean(DEC_axis))))>1.5):
        printlog("making PSF for kernelsize " + str(kernel_size) + ", declination " + "{d:.2f}".format(d=np.nanmean(DEC_axis)),output_file=processfile)
        default_PSF = scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),kernel_size//2,True,nsamps)
        default_PSF_params = (kernel_size,float("{d:.2f}".format(d=np.nanmean(DEC_axis))))


    canddict = dict()

    #print("starting process " + str(img_id) + "...")
    timing1 = time.time()
    if PyTorchDedispersion: #uses Nikita's dedisp code
        total_noise = None
        printlog("Using PyTorchDedispersion",output_file=processfile)
        fullimg.candidxs,fullimg.cands,fullimg.image_tesseract_searched,fullimg.image_tesseract_binned,canddict,tmp = run_PyTorchDedisp_search(fullimg.image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,SNRthresh=SNRthresh,canddict=dict(),output_file=output_file,usefft=usefft,space_filter=space_filter,noiseth=noiseth,RA_cutoff=0 if nocutoff else get_RA_cutoff(fullimg.DEC_axis[len(fullimg.DEC_axis)//2],pixsize=fullimg.DEC_axis[1]-fullimg.DEC_axis[0]))

    else:
        fullimg.candidxs,fullimg.cands,fullimg.image_tesseract_searched,fullimg.image_tesseract_binned,canddict,tmp,tmp,tmp,tmp,total_noise = run_search_new(fullimg.image_tesseract,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,canddict=dict(),usefft=usefft,multithreading=multithreading,nrows=nrows,ncols=ncols,output_file=output_file,threadDM=threadDM,samenoise=samenoise,cuda=cuda,space_filter=space_filter,kernel_size=kernel_size,exportmaps=exportmaps,append_frame=append_frame,DMbatches=DMbatches,SNRbatches=SNRbatches,usejax=usejax,noiseth=noiseth,RA_cutoff=0 if nocutoff else get_RA_cutoff(fullimg.DEC_axis[len(fullimg.DEC_axis)//2],pixsize=fullimg.DEC_axis[1]-fullimg.DEC_axis[0]))


    #update noise stats
    if total_noise is not None:
        global current_noise
        current_noise = (noise_update_all(total_noise,gridsize,gridsize,DM_trials,widthtrials),current_noise[1] + 1)

    #update last frame
    if append_frame:
        global last_frame
        save_last_frame(last_frame,full=True)
        printlog("Writing to last_frame.npy",output_file=processfile)

    if savesearch or len(fullimg.candidxs)>0:
        #write raw candidates to csv
        csvfile = open(cand_dir + "raw_cands/candidates_" + fullimg.img_id_isot + ".csv","w")
        wr = csv.writer(csvfile,delimiter=',')
        wr.writerow(["candname","RA index","DEC index","WIDTH index", "DM index", "SNR"])
        for i in range(len(fullimg.candidxs)):
            wr.writerow(np.concatenate([[i],np.array(fullimg.candidxs[i][:-1],dtype=int),[fullimg.candidxs[i][-1]]]))
        csvfile.close()

        #save image
        f = open(cand_dir + "raw_cands/" + fullimg.img_id_isot + ".npy","wb")
        np.save(f,fullimg.image_tesseract_binned)
        f.close()

        #save fits
        numpy_to_fits(fullimg.image_tesseract_binned.astype(np.float32),cand_dir + "raw_cands/" + fullimg.img_id_isot + ".fits")

        #save image
        f = open(cand_dir + "raw_cands/" + fullimg.img_id_isot + "_searched.npy","wb")
        np.save(f,fullimg.image_tesseract_searched)
        f.close()
        
        #save fits
        numpy_to_fits(fullimg.image_tesseract_searched.astype(np.float32),cand_dir + "raw_cands/" + fullimg.img_id_isot + "_searched.fits")


        #if the dask scheduler is set up, put the cand file name in the queue
        #if 'DASKPORT' in os.environ.keys() and QSETUP:
        #    QQUEUE.put("candidates_" + fullimg.img_id_isot + ".csv")

    printlog(fullimg.image_tesseract_searched,output_file=processfile)
    printlog("done, total search time: " + str(np.around(time.time()-timing1,2)) + " s",output_file=processfile)

    if len(fullimg.candidxs)==0:
        printlog("No candidates found",output_file=processfile)
        return fullimg.image_tesseract_searched,None#fullimg.cands,fullimg.candidxs,len(fullimg.cands)
    else:
        printlog(str(len(fullimg.candidxs)) + " candidates found",output_file=processfile)
        return fullimg.image_tesseract_searched,"candidates_" + fullimg.img_id_isot + ".csv"



#get cands and clusters from csv file
def read_cands(fname):
    cands = []
    with open(cand_dir+fname,"r") as csvfile:
        rdr = csv.reader(csvfile,delimiter=',')
        for row in rdr:#.read_row():
            cands.append(row)
    csvfile.close()
    return cands



import pickle as pkl
#function to update the currently stored noisemap
def update_noisemap(image_tesseract,noisemap_file="/dataz/dsa110/imaging/NSFRB_storage/NSFRB_noisemaps/noisemap_256pix.npy",noisemap_info_file="/dataz/dsa110/imaging/NSFRB_storage/NSFRB_noisemaps/noisemap_256pix_info.pkl"):
    #get previous noise map
    noisemap = np.load(noisemap_file)

    #get info from noise map
    f = open(noisemap_info_file,'rb')
    noiseinfo = pkl.load(f)
    f.close()

    #get standard deviation of new image
    imagenoisemap = image_tesseract.std(2)
    imagenoisemap = np.expand_dims(imagenoisemap,axis=2)
    
    if noiseinfo['iterations'] == 0:#np.all(noisemap == -1):
        #if all -1, replace noise map
        newnoisemap = imagenoisemap
    else:
        #otherwise, average together standard deviations
        newnoisemap = (noisemap*noiseinfo['iterations'] + imagenoisemap)/(noiseinfo['iterations'] + 1)

    #increase iterations
    noiseinfo['iterations'] += 1

    #write to file
    np.save(noisemap_file,newnoisemap)
    f = open(noisemap_info_file,'wb')
    pkl.dump(noiseinfo,f)
    f.close()
    return newnoisemap


#normalize using noisemap
def normalize_image(image_tesseract,noisemap_file="/dataz/dsa110/imaging/NSFRB_storage/NSFRB_noisemaps/noisemap_256pix.npy",noisemap_info_file="/dataz/dsa110/imaging/NSFRB_storage/NSFRB_noisemaps/noisemap_256pix_info.pkl",output_file=output_file):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #get previous noise map
    noisemap = np.load(noisemap_file)

    #get info from noise map
    f = open(noisemap_info_file,'rb')
    noiseinfo = pkl.load(f)
    f.close()

    if noiseinfo['iterations']==0:
        print("Noise map not initialized",file=fout)
        if output_file != "":
            fout.close()
        return image_tesseract
    else:
        if output_file != "":
            fout.close()
        return image_tesseract/noisemap

"""
#code to cutout subimages
def get_subimage(image_tesseract,ra_idx,dec_idx,dm=-1,freq_axis=freq_axis,tsamp=130,subimgpix=11,save=False,prefix="candidate_stamp",plot=False,output_file=output_file,output_dir=cand_dir):
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    gridsize = image_tesseract.shape[0]
    fname = output_dir + prefix + "_" + str(ra_idx) + "_" + str(dec_idx)
    if subimgpix%2 == 0:
        print("subimgpix must be odd",file=fout)
        if output_file != "":
            fout.close()
        return None



    #dedisperse if given a dm
    if dm != -1:
        fname = fname + "_dedisp" + str(dm) + ".npy"
        image_tesseract_dm = copy.deepcopy(image_tesseract)
        for i in range(gridsize):
            for j in range(gridsize):
                tdelays = dm*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
                tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
                tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
                tdelays_frac = tdelays/tsamp - tdelays_idx_low

                for k in range(nchans):
                    #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac)
                    arrlow =  np.pad(image_tesseract[i,j,:,k],((0,tdelays_idx_low[k])),mode="constant",constant_values=0)[tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                    arrhi =  np.pad(image_tesseract[i,j,:,k],((0,tdelays_idx_hi[k])),mode="constant",constant_values=0)[tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

                    image_tesseract_dm[i,j,:,k] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])

    else:
        fname = fname + ".npy"
        image_tesseract_dm = copy.deepcopy(image_tesseract)

    #pad w/ nans
    image_tesseract_dm = np.pad(image_tesseract_dm,((gridsize,gridsize),
                                                   (gridsize,gridsize),
                                                   (0,0),
                                                   (0,0)),
                                                   mode='constant',
                                                   constant_values=np.nan)

    #cut out subimage
    minraidx = int(gridsize + ra_idx - subimgpix//2)#np.max([ra_idx - subimgpix//2,0])
    maxraidx = int(gridsize + ra_idx + subimgpix//2 + 1)#np.min([ra_idx + subimgpix//2 + 1,gridsize-1])
    mindecidx = int(gridsize + dec_idx - subimgpix//2)#np.max([dec_idx - subimgpix//2,0])
    maxdecidx = int(gridsize + dec_idx + subimgpix//2 + 1)#np.min([dec_idx + subimgpix//2 + 1,gridsize-1])

    #print(minraidx_cut,maxraidx_cut,mindecidx_cut,maxdecidx_cut)
    print(minraidx,maxraidx,mindecidx,maxdecidx,file=fout)

    image_cutout = image_tesseract_dm[minraidx:maxraidx,mindecidx:maxdecidx,:,:]

    if save:
        np.save(fname,image_cutout)

    if plot:
        plt.figure(figsize=(12,12))
        plt.imshow(image_cutout.mean((2,3)),aspect='auto')
        plt.show()
    if output_file != "":
        fout.close()
    return image_cutout





#hdbscan clustering function; clusters in DM, width, RA, DEC space
import hdbscan
def hdbscan_cluster(cands,min_cluster_size=50,gridsize=gridsize,nDMtrials=nDMtrials,nwidths=nwidths,dmt=DM_trials,wt=widthtrials,SNRthresh=SNRthresh,plot=False,show=False):

    print(str(len(cands)) + " candidates")

    #make list for each param
    raidxs = []
    decidxs = []
    dmidxs = []
    widthidxs = []
    snridxs = []
    for i in range(len(cands)):
        raidxs.append(cands[i][0])
        decidxs.append(cands[i][1])
        dmidxs.append(cands[i][3])
        widthidxs.append(cands[i][2])
        snridxs.append(cands[i][4])
    raidxs = np.array(raidxs)
    decidxs = np.array(decidxs)
    dmidxs = np.array(dmidxs)
    widthidxs = np.array(widthidxs)
    snridxs = np.array(snridxs)

    test_data=np.array([raidxs,decidxs,dmidxs,widthidxs]).transpose()


    #create clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)

    #cluster data
    clusterer.fit(test_data)


    #print number of noise points
    noisepoints = np.sum(clusterer.labels_==-1)
    print(str(noisepoints) + " noise points")

    nclasses = len(np.unique(clusterer.labels_))
    classnames = np.unique(clusterer.labels_)
    classes = clusterer.labels_
    if -1 in clusterer.labels_:
        nclasses -= 1

    print(str(nclasses) + " unique classes")


    #get centroids
    fcsv = open(cand_dir + "hdbscan_cluster_cands.csv","w")
    csvwriter = csv.writer(fcsv)
    centroid_ras = []
    centroid_decs = []
    centroid_dms = []
    centroid_widths = []
    centroid_snrs = []
    for k in classnames:
        if k != -1:
            centroid_ras.append((np.nansum((snridxs*raidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_decs.append((np.nansum((snridxs*decidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_dms.append((np.nansum((snridxs*dmidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_widths.append((np.nansum((snridxs*widthidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_snrs.append(np.nansum((snridxs*snridxs)[classes==k])/np.nansum(snridxs[classes==k]))
            csvwriter.writerow([centroid_ras[-1],centroid_decs[-1],centroid_widths[-1],centroid_dms[-1],centroid_snrs[-1]])            
    fcsv.close()
    centroid_ras = np.array(centroid_ras)
    centroid_decs = np.array(centroid_decs)
    centroid_dms = np.array(centroid_dms)
    centroid_widths = np.array(centroid_widths)
    centroid_snrs = np.array(centroid_snrs)

    centroid_cands = [(centroid_ras[i],centroid_decs[i],centroid_widths[i],centroid_dms[i],centroid_snrs[i]) for i in range(len(centroid_ras))]

    if plot:
        cands_noninf = []
        for i in cands:
            if not np.isinf(i[-1]): cands_noninf.append(i)
        
        plt.figure(figsize=(40,12))
        plt.subplot(121)
        for i in range(-1,len(np.unique(classes))-int(-1 in classes)):
            if i == -1:
                plt.scatter(np.array(cands_noninf)[classes==i,0],np.array(cands_noninf)[classes==i,1],alpha=0.5,s=1000*(np.array(cands_noninf)[classes==i,-1] - SNRthresh)/(2*SNRthresh - SNRthresh),label='Not Classified',color='grey')
            else:
                c=plt.plot(centroid_ras[i],centroid_decs[i],'x',markersize=50,markerfacecolor="none",markeredgewidth=4)
                plt.scatter(np.array(cands_noninf)[classes==i,0],np.array(cands_noninf)[classes==i,1],alpha=0.5,s=1000*(np.array(cands_noninf)[classes==i,-1] - SNRthresh)/(2*SNRthresh - SNRthresh),label='Class ' + str(i),c=c[0].get_color())

        plt.xlim(0,32)
        plt.ylim(0,32)
        plt.xlabel("RA index")
        plt.ylabel("DEC index")
        plt.legend(loc='upper right')

        plt.subplot(122)
        wtinterp = interp1d(np.arange(len(wt)),wt,fill_value='extrapolate')
        dmtinterp = interp1d(np.arange(len(dmt)),dmt,fill_value='extrapolate')
        for i in range(-1,len(np.unique(classes))-int(-1 in classes)):
            if i == -1:
                plt.scatter(wt[np.array(cands_noninf,dtype=int)[classes==i,2]],dmt[np.array(cands_noninf,dtype=int)[classes==i,3]],alpha=0.5,s=1000*(np.array(cands_noninf)[classes==i,-1] - SNRthresh)/(2*SNRthresh - SNRthresh),label='Not Classified',color='grey')
            else:
                c=plt.plot(int(wtinterp(centroid_widths[i])),int(dmtinterp(centroid_dms[i])),'x',markersize=50,markerfacecolor="none",markeredgewidth=4)
                plt.scatter(wt[np.array(cands_noninf,dtype=int)[classes==i,2]],dmt[np.array(cands_noninf,dtype=int)[classes==i,3]],alpha=0.5,s=1000*(np.array(cands_noninf)[classes==i,-1] - SNRthresh)/(2*SNRthresh - SNRthresh),label='Class ' + str(i),c=c[0].get_color())
        plt.xlabel("Width (Samples)")
        plt.ylabel("DM (pc/cc)")
        plt.legend(loc='upper right',frameon=True)
	

        plt.savefig(cand_dir + "hdbscan_cluster_plot.png")
        if show:        
            plt.show()
        else:
            plt.close()

    return classes,centroid_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs
"""

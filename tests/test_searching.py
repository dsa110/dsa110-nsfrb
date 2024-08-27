import numpy as np
import jax
import torch
import socket
import time
from matplotlib import pyplot as plt
import random
import copy
from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import truncnorm
from scipy.signal import peak_widths
from scipy.stats import norm
#from event import names
#from gen_dmtrials_copy import gen_dm
import argparse
from astropy.time import Time
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor


import os
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
cwd = os.environ['NSFRBDIR']
import sys
sys.path.append(cwd + "/") #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
import csv
import copy

from nsfrb.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset
from nsfrb import searching as sl
from nsfrb import pipeline
from nsfrb.outputlogging import printlog

"""
This script runs a series of unit tests to verify the NSFRB search pipeline
"""


def base_test(img,PSF,
        SNRthresh,
        gridsize,
        nsamps,
        nchans,verbose,usefft=False,multithreading=False,threadDM=False,cuda=False):
    """
    Run search pipeline 
    """
    if not (usefft or multithreading): print("Testing Baseline Search Pipeline...",end="")
    elif usefft and not multithreading: print("Testing Search Pipeline with FFT implementation...",end="")
    elif multithreading and (not threadDM) and (not usefft): print("Testing Search Pipeline with Multithreading (No DM Threading)...",end="")
    elif multithreading and threadDM and (not usefft): print("Testing Search Pipeline with Multithreading Implementation and DM Threading...",end="")
    elif usefft and multithreading and (not threadDM): print("Testing Search Pipeline with FFT and Multithreading (No DM Threading)...",end="")
    elif usefft and multithreading and threadDM: print("Testing Search Pipeline with FFT and Multithreading Implementation and DM Threading...",end="")
    print("yololol",gridsize,img.shape,PSF.shape)
    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    RA_axis = np.linspace(-gridsize//2,gridsize//2,gridsize)
    DEC_axis=np.linspace(-gridsize//2,gridsize//2,gridsize)
    time_axis = np.arange(nsamps)*sl.tsamp
    if verbose: ofile = ""
    else: ofile = sl.output_file
    candidxs,cands,image_tesseract_searched,image_tesseract_binned,canddict,tmp,tmp,tmp,tmp = sl.run_search_new(img,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                                                                                    time_axis=time_axis,canddict=dict(),
                                                                                                    PSF=PSF,output_file=ofile,
                                                                                                    usefft=usefft,multithreading=multithreading,threadDM=threadDM,cuda=cuda)
                                                                                                    
    """
    ASSERTIONS
    """
    assert(image_tesseract_searched.shape == (gridsize,gridsize,len(sl.widthtrials),len(sl.DM_trials)))
    assert(image_tesseract_binned.shape == (gridsize,gridsize,nsamps,nchans))
    assert(type(candidxs) == list)
    assert(type(cands) == list)
    assert(len(candidxs) == len(cands))
    for k in canddict.keys():
        assert(len(canddict[k]) == len(candidxs))
    print("Passed!")                     
    return


def lowSNR_test(img,PSF,
        gridsize,
        nsamps,
        nchans,verbose,usefft=False,multithreading=False,threadDM=False,cuda=False):
    """
    Run search pipeline with low SNR threshold
    """

    if not (usefft or multithreading): print("Testing Baseline Search Pipeline with low SNR threshold...",end="")
    elif usefft and not multithreading: print("Testing Search Pipeline with FFT implementation with low SNR threshold...",end="")
    elif multithreading and (not threadDM) and (not usefft): print("Testing Search Pipeline with Multithreading (No DM Threading) with low SNR threshold...",end="")
    elif multithreading and threadDM and (not usefft): print("Testing Search Pipeline with Multithreading Implementation and DM Threading with low SNR threshold...",end="")
    elif usefft and multithreading and (not threadDM): print("Testing Search Pipeline with FFT and Multithreading (No DM Threading) with low SNR threshold...",end="")
    elif usefft and multithreading and threadDM: print("Testing Search Pipeline with FFT and Multithreading Implementation and DM Threading with low SNR threshold...",end="")


    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    RA_axis = np.linspace(-gridsize//2,gridsize//2,gridsize)
    DEC_axis=np.linspace(-gridsize//2,gridsize//2,gridsize)
    time_axis = np.arange(nsamps)*sl.tsamp
    if verbose: ofile = ""
    else: ofile = sl.output_file
    SNRthresh = 0
    candidxs,cands,image_tesseract_searched,image_tesseract_binned,canddict,tmp,tmp,tmp,tmp = sl.run_search_new(img,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                                                                                    time_axis=time_axis,canddict=dict(),
                                                                                                    PSF=PSF,output_file=ofile,
                                                                                                    usefft=usefft,multithreading=multithreading,threadDM=threadDM,cuda=cuda)
    """
    ASSERTIONS
    """
    assert(image_tesseract_searched.shape == (gridsize,gridsize,len(sl.widthtrials),len(sl.DM_trials)))
    assert(image_tesseract_binned.shape == (gridsize,gridsize,nsamps,nchans))
    assert(type(candidxs) == list)
    assert(type(cands) == list)
    assert(len(candidxs) == len(cands))
    for k in canddict.keys():
        assert(len(canddict[k]) == len(candidxs))
    assert(len(candidxs) == np.sum(~np.isnan(image_tesseract_searched)))
    #assert(len(candidxs) == np.sum(np.logical_and(image_tesseract_searched > SNRthresh,~np.isnan(image_tesseract_searched))))# == gridsize*gridsize*len(sl.widthtrials)*len(sl.DM_trials)) #check that all positions, DMs, widths are recovered 
    assert(len(candidxs[0]) == 5) #RA, DEC, width, DM,SNR
    assert(len(cands[0]) == 5) #RA, DEC, width, DM, SNR
    assert(np.all(canddict['ras'] == RA_axis[canddict['ra_idxs']]))
    assert(np.all(canddict['decs'] == DEC_axis[canddict['dec_idxs']]))
    assert(np.all(canddict['wids'] == sl.widthtrials[canddict['wid_idxs']]))
    assert(np.all(canddict['dms'] == sl.DM_trials[canddict['dm_idxs']]))
    assert(np.all(np.array([candidxs[i][0] for i in range(len(candidxs))]) == canddict['ra_idxs']))
    assert(np.all(np.array([candidxs[i][1] for i in range(len(candidxs))]) == canddict['dec_idxs']))
    assert(np.all(np.array([candidxs[i][2] for i in range(len(candidxs))]) == canddict['wid_idxs']))
    assert(np.all(np.array([candidxs[i][3] for i in range(len(candidxs))]) == canddict['dm_idxs']))
    assert(np.all(np.array([cands[i][0] for i in range(len(cands))]) == canddict['ras']))
    assert(np.all(np.array([cands[i][1] for i in range(len(cands))]) == canddict['decs']))
    assert(np.all(np.array([cands[i][2] for i in range(len(cands))]) == canddict['wids']))
    assert(np.all(np.array([cands[i][3] for i in range(len(cands))]) == canddict['dms']))
    assert(np.all(np.array([cands[i][4] for i in range(len(cands))]) == np.array([candidxs[i][4] for i in range(len(candidxs))])))
    assert(np.all(np.array([cands[i][4] for i in range(len(cands))]) == np.array([image_tesseract_searched[candidxs[i][1],candidxs[i][0],candidxs[i][2],candidxs[i][3]] for i in range(len(cands))])))
    print("Passed!")
    return


def highSNR_test(img,PSF,
        SNRthresh,
        gridsize,
        nsamps,
        nchans,verbose,usefft=False,multithreading=False,threadDM=False,cuda=False):
    """
    Run search pipeline with high SNR threshold
    """
    
    if not (usefft or multithreading): print("Testing Baseline Search Pipeline with high SNR threshold...",end="")
    elif usefft and not multithreading: print("Testing Search Pipeline with FFT implementation with high SNR threshold...",end="")
    elif multithreading and (not threadDM) and (not usefft): print("Testing Search Pipeline with Multithreading (No DM Threading) with high SNR threshold...",end="")
    elif multithreading and threadDM and (not usefft): print("Testing Search Pipeline with Multithreading Implementation and DM Threading with high SNR threshold...",end="")
    elif usefft and multithreading and (not threadDM): print("Testing Search Pipeline with FFT and Multithreading (No DM Threading) with high SNR threshold...",end="")
    elif usefft and multithreading and threadDM: print("Testing Search Pipeline with FFT and Multithreading Implementation and DM Threading with high SNR threshold...",end="")


    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    RA_axis = np.linspace(-gridsize//2,gridsize//2,gridsize)
    DEC_axis=np.linspace(-gridsize//2,gridsize//2,gridsize)
    time_axis = np.arange(nsamps)*sl.tsamp
    if verbose: ofile = ""
    else: ofile = sl.output_file
    candidxs,cands,image_tesseract_searched,image_tesseract_binned,canddict,tmp,tmp,tmp,tmp = sl.run_search_new(img,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                                                                                    time_axis=time_axis,canddict=dict(),
                                                                                                    PSF=PSF,output_file=ofile,
                                                                                                    usefft=usefft,multithreading=multithreading,threadDM=threadDM,cuda=cuda)
    """
    ASSERTIONS
    """
    assert(image_tesseract_searched.shape == (gridsize,gridsize,len(sl.widthtrials),len(sl.DM_trials)))
    assert(image_tesseract_binned.shape == (gridsize,gridsize,nsamps,nchans))
    assert(type(candidxs) == list)
    assert(type(cands) == list)
    assert(len(candidxs) == len(cands))
    for k in canddict.keys():
        assert(len(canddict[k]) == len(candidxs))
    assert(len(candidxs) == 0)
    assert(len(cands) == 0)
    assert(np.all(image_tesseract_searched[~np.isnan(image_tesseract_searched)] <= SNRthresh))
    print("Passed!")
    return




#regular implementation
def test_regular_implementation():


    SNRthresh = 3000
    gridsize = 32
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False
    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)
    base_test(img,PSFimg,SNRthresh,gridsize,nsamps,nchans,verbose)
    lowSNR_test(img,PSFimg,gridsize,nsamps,nchans,verbose)
    highSNR_test(img,PSFimg,10000,gridsize,nsamps,nchans,verbose)

    return



#FFT implementation
def test_FFT_implementation():


    SNRthresh = 3000
    gridsize = 32
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False
    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)
    base_test(img,PSFimg,SNRthresh,gridsize,nsamps,nchans,verbose,usefft=True)
    lowSNR_test(img,PSFimg,gridsize,nsamps,nchans,verbose,usefft=True)
    highSNR_test(img,PSFimg,10000,gridsize,nsamps,nchans,verbose,usefft=True)

    return


#GPU accelerated implementation
def test_GPU_implementation():

    SNRthresh = 3000
    gridsize = 32
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False
    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)
    base_test(img,PSFimg,SNRthresh,gridsize,nsamps,nchans,verbose,usefft=False,cuda=True)
    lowSNR_test(img,PSFimg,gridsize,nsamps,nchans,verbose,usefft=False,cuda=True)
    highSNR_test(img,PSFimg,10000,gridsize,nsamps,nchans,verbose,usefft=False,cuda=True)

    return


#GPU accelerated implementation with FFT
def test_FFT_GPU_implementation():

    SNRthresh = 3000
    gridsize = 32
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False
    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)

    base_test(img,PSFimg,SNRthresh,gridsize,nsamps,nchans,verbose,usefft=True,cuda=True)
    lowSNR_test(img,PSFimg,gridsize,nsamps,nchans,verbose,usefft=True,cuda=True)
    highSNR_test(img,PSFimg,10000,gridsize,nsamps,nchans,verbose,usefft=True,cuda=True)

    return



#multithreading implementation
def test_multithreading_implementation():

    SNRthresh = 3000
    gridsize = 32
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False

    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)

    base_test(img,PSFimg,SNRthresh,gridsize,nsamps,nchans,verbose,multithreading=True)
    lowSNR_test(img,PSFimg,gridsize,nsamps,nchans,verbose,multithreading=True)
    highSNR_test(img,PSFimg,10000,gridsize,nsamps,nchans,verbose,multithreading=True)
    
    return




def test_boxcar_filter():
    SNRthresh = 3000
    gridsize = 300
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False
    batches = 4

    sl.init_last_frame(gridsize,gridsize,nsamps,nchans)
    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)

    device = torch.device(random.choice(np.arange(torch.cuda.device_count(),dtype=int)) if torch.cuda.is_available() else "cpu")


    imgout = sl.snr_vs_RA_DEC_allDMW(torch.from_numpy(img),DM_trials=sl.DM_trials,widthtrials=sl.widthtrials,mode='4d',noiseth=0.9,samenoise=True,plot=False,device=device,usefft=True,batches=batches,usejax=True,maxProcesses=5)
    imgout = sl.snr_vs_RA_DEC_allDMW(torch.from_numpy(img),DM_trials=sl.DM_trials,widthtrials=sl.widthtrials,mode='4d',noiseth=0.9,samenoise=True,plot=False,device=device,usefft=True,batches=batches,usejax=True,maxProcesses=5)
    imgout = sl.snr_vs_RA_DEC_allDMW(torch.from_numpy(img),DM_trials=sl.DM_trials,widthtrials=sl.widthtrials,mode='4d',noiseth=0.9,samenoise=True,plot=False,device=device,usefft=True,batches=batches,usejax=True,maxProcesses=5)
    imgout = sl.snr_vs_RA_DEC_allDMW(torch.from_numpy(img),DM_trials=sl.DM_trials,widthtrials=sl.widthtrials,mode='4d',noiseth=0.9,samenoise=True,plot=False,device=device,usefft=True,batches=batches,usejax=True,maxProcesses=5)
    return


from nsfrb import jax_funcs
from concurrent.futures import ThreadPoolExecutor
def test_jit_all():

    SNRthresh = 3000
    gridsize = 300
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False
    DMbatches = 3
    maxshift = 24
    tDM_max = (4.15)*np.max(sl.DM_trials)*((1/sl.fmin/1e-3)**2 - (1/sl.fmax/1e-3)**2) #ms
    maxshift = int(np.ceil(tDM_max/sl.tsamp))

    subgridsize_RA = subgridsize_DEC = gridsize//DMbatches
    executor = ThreadPoolExecutor(DMbatches*DMbatches)
    tasks = []
    for i in range(DMbatches):
        for j in range(DMbatches):
            
            tasks.append(executor.submit(jax_funcs.dedisp_snr_fft_jit_0,jax.device_put(np.array(np.random.normal(size=(gridsize//DMbatches,gridsize//DMbatches,maxshift + nsamps,nchans)),dtype=np.float32),jax.devices()[0]),jax.device_put(sl.corr_shifts_all_append[j*subgridsize_DEC:(j+1)*subgridsize_DEC,i*subgridsize_RA:(i+1)*subgridsize_RA,:,:,:],jax.devices()[0]),jax.device_put(sl.tdelays_frac_append[j*subgridsize_DEC:(j+1)*subgridsize_DEC,i*subgridsize_RA:(i+1)*subgridsize_RA,:,:,:],jax.devices()[0]),jax.device_put(np.array(np.random.normal(size=(len(sl.widthtrials),gridsize//DMbatches,gridsize//DMbatches,nsamps,len(sl.DM_trials))),dtype=np.float16),jax.devices()[0]),jax.device_put(np.array(np.random.normal(size=(len(sl.widthtrials),len(sl.DM_trials))),dtype=np.float16),jax.devices()[0]),past_noise_N=1,noiseth=0.1,i=i,j=j))
            tasks.append(executor.submit(jax_funcs.dedisp_snr_fft_jit_0,jax.device_put(np.array(np.random.normal(size=(gridsize//DMbatches,gridsize//DMbatches,maxshift + nsamps,nchans)),dtype=np.float32),jax.devices()[1]),jax.device_put(sl.corr_shifts_all_append[j*subgridsize_DEC:(j+1)*subgridsize_DEC,i*subgridsize_RA:(i+1)*subgridsize_RA,:,:,:],jax.devices()[1]),jax.device_put(sl.tdelays_frac_append[j*subgridsize_DEC:(j+1)*subgridsize_DEC,i*subgridsize_RA:(i+1)*subgridsize_RA,:,:,:],jax.devices()[1]),jax.device_put(np.array(np.random.normal(size=(len(sl.widthtrials),gridsize//DMbatches,gridsize//DMbatches,nsamps,len(sl.DM_trials))),dtype=np.float16),jax.devices()[1]),jax.device_put(np.array(np.random.normal(size=(len(sl.widthtrials),len(sl.DM_trials))),dtype=np.float16),jax.devices()[1]),past_noise_N=1,noiseth=0.1,i=i,j=j))
            """
            tasks.append(executor.submit(jax_funcs.dedisp_snr_fft_jit_0,np.array(np.random.normal(size=(gridsize//DMbatches,gridsize//DMbatches,maxshift + nsamps,nchans)),dtype=np.float32),sl.DM_trials,sl.tsamp,sl.freq_axis,np.array(np.random.normal(size=(len(sl.widthtrials),gridsize//DMbatches,gridsize//DMbatches,nsamps,len(sl.DM_trials))),dtype=np.float16),np.array(np.random.normal(size=(len(sl.widthtrials),len(sl.DM_trials))),dtype=np.float16),past_noise_N=1,noiseth=0.1,i=i,j=j))
            tasks.append(executor.submit(jax_funcs.dedisp_snr_fft_jit_1,np.array(np.random.normal(size=(gridsize//DMbatches,gridsize//DMbatches,maxshift + nsamps,nchans)),dtype=np.float32),sl.DM_trials,sl.tsamp,sl.freq_axis,np.array(np.random.normal(size=(len(sl.widthtrials),gridsize//DMbatches,gridsize//DMbatches,nsamps,len(sl.DM_trials))),dtype=np.float16),np.array(np.random.normal(size=(len(sl.widthtrials),len(sl.DM_trials))),dtype=np.float16),past_noise_N=1,noiseth=0.1,i=i,j=j))
            """
    for t in tasks:
        res = t.result()
    executor.shutdown()
    return


if __name__=="__main__":
    pytest.main()


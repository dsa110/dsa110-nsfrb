import numpy as np
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
from event import names
#from gen_dmtrials_copy import gen_dm
import argparse
from astropy.time import Time
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()

import os
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
    assert(len(candidxs) == gridsize*gridsize*len(sl.widthtrials)*len(sl.DM_trials)) #check that all positions, DMs, widths are recovered 
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
    assert(np.all(image_tesseract_searched <= SNRthresh))
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

    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)

    base_test(img,PSFimg,SNRthresh,gridsize,nsamps,nchans,verbose,multithreading=True)
    lowSNR_test(img,PSFimg,gridsize,nsamps,nchans,verbose,multithreading=True)
    highSNR_test(img,PSFimg,10000,gridsize,nsamps,nchans,verbose,multithreading=True)
    
    return


#multithreading and DM threading
def test_mulithreading_with_DM_threading():
    SNRthresh = 3000
    gridsize = 32
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False

    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)

    base_test(img,PSFimg,SNRthresh,gridsize,nsamps,nchans,verbose,multithreading=True,threadDM=True)
    lowSNR_test(img,PSFimg,gridsize,nsamps,nchans,verbose,multithreading=True,threadDM=True)
    highSNR_test(img,PSFimg,10000,gridsize,nsamps,nchans,verbose,multithreading=True,threadDM=True)

    return

#FFT, multithreading, DM threading
def test_FFT_and_multithreading_with_DM_threading():
    SNRthresh = 3000
    gridsize = 32
    nsamps = 25
    nchans =  16
    ofile = sl.output_file
    verbose = False

    PSFimg = sl.make_PSF_cube(gridsize=gridsize,nsamps=nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=gridsize,nsamps=nsamps,DM=0,output_file=ofile)

    base_test(img,PSFimg,SNRthresh,gridsize,nsamps,nchans,verbose,multithreading=True,threadDM=True,usefft=True)
    lowSNR_test(img,PSFimg,gridsize,nsamps,nchans,verbose,multithreading=True,threadDM=True,usefft=True)
    highSNR_test(img,PSFimg,10000,gridsize,nsamps,nchans,verbose,multithreading=True,threadDM=True,usefft=True)

    return


"""
def main():
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 3000',default=3000)
    parser.add_argument('--port',type=int,help='Port number for receiving data from subclient, default = 8843',default=8843)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, default=300',default=300)
    parser.add_argument('--nsamps',type=int,help='Expected number of time samples (integrations) for each sub-band image, default=25',default=25)
    parser.add_argument('--nchans',type=int,help='Expected number of sub-band images for each full image, default=16',default=16)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--usefft',action='store_true', help='Implement PSF spatial matched filter as a 2D FFT')
    parser.add_argument('--cluster',action='store_true',help='Enable clustering with HDBSCAN')
    parser.add_argument('--multithreading',action='store_true',help='Enable multithreading in search')
    parser.add_argument('--nrows',type=int,help='Number of rows to break image into if multithreading, default = 4',default=4)
    parser.add_argument('--ncols',type=int,help='Number of columns to break image into if multithreading, default = 2',default=2)
    parser.add_argument('--threadDM',action='store_true',help='Break DM trials among multiple threads')
    parser.add_argument('--run_unit_tests',action='store_true',help='Run all unit tests with set parameters')
    args = parser.parse_args()



    #create a test image and PSF
    if args.verbose: ofile = ""
    else: ofile = sl.output_file
    print("Creating test images and PSF...",end="")
    PSFimg = sl.make_PSF_cube(gridsize=args.gridsize,nsamps=args.nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=args.gridsize,nsamps=args.nsamps,DM=0,output_file=ofile)
    print("Done!")

    #TESTING
    if args.run_unit_tests:
        #regular implementation
        test_1(img,PSFimg,args.SNRthresh,args.gridsize,args.nsamps,args.nchans,args.verbose)
        test_2(img,PSFimg,args.gridsize,args.nsamps,args.nchans,args.verbose)
        test_3(img,PSFimg,10000,args.gridsize,args.nsamps,args.nchans,args.verbose)

        #with FFT
        test_1(img,PSFimg,args.SNRthresh,args.gridsize,args.nsamps,args.nchans,args.verbose,usefft=True)
        test_2(img,PSFimg,args.gridsize,args.nsamps,args.nchans,args.verbose,usefft=True)
        test_3(img,PSFimg,10000,args.gridsize,args.nsamps,args.nchans,args.verbose,usefft=True)

        #with multithreading
        test_1(img,PSFimg,args.SNRthresh,args.gridsize,args.nsamps,args.nchans,args.verbose,multithreading=True)
        test_2(img,PSFimg,args.gridsize,args.nsamps,args.nchans,args.verbose,multithreading=True)
        test_3(img,PSFimg,10000,args.gridsize,args.nsamps,args.nchans,args.verbose,multithreading=True)
        
        #with multithreading and DM threading
        test_1(img,PSFimg,args.SNRthresh,args.gridsize,args.nsamps,args.nchans,args.verbose,multithreading=True,threadDM=True)
        test_2(img,PSFimg,args.gridsize,args.nsamps,args.nchans,args.verbose,multithreading=True,threadDM=True)
        test_3(img,PSFimg,10000,args.gridsize,args.nsamps,args.nchans,args.verbose,multithreading=True,threadDM=True)

        #with FFT, multithreading, and DM threading
        test_1(img,PSFimg,args.SNRthresh,args.gridsize,args.nsamps,args.nchans,args.verbose,usefft=True,multithreading=True,threadDM=True)
        test_2(img,PSFimg,args.gridsize,args.nsamps,args.nchans,args.verbose,usefft=True,multithreading=True,threadDM=True)
        test_3(img,PSFimg,10000,args.gridsize,args.nsamps,args.nchans,args.verbose,usefft=True,multithreading=True,threadDM=True)
    return
"""
if __name__=="__main__":
    pytest.main()


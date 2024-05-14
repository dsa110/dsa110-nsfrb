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


def test_1(img,PSF,
        SNRthresh,
        gridsize,
        nsamps,
        nchans,verbose):
    """
    Run search pipeline without multithreading or FFT or clustering
    """
    RA_axis = np.linspace(-gridsize//2,gridsize//2,gridsize)
    DEC_axis=np.linspace(-gridsize//2,gridsize//2,gridsize)
    time_axis = np.arange(nsamps)*sl.tsamp
    if verbose: ofile = ""
    else: ofile = sl.output_file
    candidxs,cands,image_tesseract_searched,image_tesseract_binned,canddict,tmp = sl.run_search_new(img,SNRthresh=SNRthresh,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                                                                                    time_axis=time_axis,canddict=dict(),
                                                                                                    PSF=PSF,output_file=ofile,
                                                                                                    usefft=False,multithreading=False)
                                                                                                    
    """
    ADD ASSERTIONS HERE
    """
    print("Passed!")                     
    return


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
    PSFimg = sl.make_PSF_cube(gridsize=args.gridsize,nsamps=args.nsamps,output_file=ofile)
    img = sl.make_image_cube(PSFimg=PSFimg,snr=1000,gridsize=args.gridsize,nsamps=args.nsamps,DM=0,output_file=ofile)
    

    #TESTING
    test_1(img,PSFimg,args.SNRthresh,args.gridsize,args.nsamps,args.nchans,args.verbose)



if __name__=="__main__":
    main()

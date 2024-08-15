import numpy as np
import csv
#from nsfrb import searching as sl
#from nsfrb.searching import make_PSF_cube,default_PSF,datagridsize
from nsfrb import simulating as sim
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
from scipy.interpolate import interp1d
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from astropy.time import Time
from nsfrb import TXclient
import argparse
from concurrent.futures import ProcessPoolExecutor

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()
sys.path.append(cwd + "/")

from nsfrb.config import *
error_file = cwd + "-logfiles/error_log.txt"
log_file = cwd + "-logfiles/inject_log.txt"
inject_file = cwd + "-injections/injections.csv"

def main():
    #redirect stderr
    sys.stderr = open(error_file,"w")


    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--SNR',type=float,help='SNR of injected burst, default = 100',default=100)
    parser.add_argument('--port',type=int,help='Port number for sending injected burst, default = ' + str(TXclient.port),default=TXclient.port)
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, default=300',default=300)
    parser.add_argument('--nsamps',type=int,help='Expected number of time samples (integrations) for each sub-band image, default=25',default=25)
    parser.add_argument('--nchans',type=int,help='Expected number of sub-band images for each full image, default=16',default=16)
    parser.add_argument('--width',type=int,help='Width of the burst in samples, default = 4',default=4)
    parser.add_argument('--DM',type=float,help='Dispersion measure of injected burst, default = 0',default=0)
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--nbursts',type=int,help='Number of injected bursts; default = 1; if > 1, the SNR, width, and DM are drawn from normal distributions centered on the provided values',default=1)
    args = parser.parse_args()

    #make image
    PSF = sim.make_PSF_cube(gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,output_file=log_file)

    if args.nbursts == 1:
        #get current time
        time_start_isot = Time.now().isot
        print("Injecting burst " + str(time_start_isot) + " with DM = " + str(args.DM) + ", width = " + str(args.width) + ", S/N = " + str(args.SNR))
        
        image_tesseract = sim.make_image_cube(PSFimg=PSF,snr=args.SNR*100,width=args.width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=args.DM,output_file=log_file)

        #send
        for i in range(nchans):#NUM_CHANNELS//AVERAGING_FACTOR):
            #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
            msg=TXclient.send_data(time_start_isot, image_tesseract[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.port)
            if args.verbose: print(msg)
            time.sleep(1)

        #report in injections file
        with open(inject_file,"a") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            wr.writerow([time_start_isot,args.DM,args.width,args.SNR])
        csvfile.close()

    else:
        SNRs = norm.rvs(loc=args.SNR,scale=1,size=args.nbursts)
        widths = np.array(np.clip(norm.rvs(loc=args.width,scale=1,size=args.nbursts),0,args.nsamps),dtype=int)
        DMs = norm.rvs(loc=args.DM,scale=args.DM/10,size=args.nbursts)
        for j in range(args.nbursts):
            SNR = SNRs[j]
            width = widths[j]
            DM = DMs[j] 

            #get current time
            time_start_isot = Time.now().isot

            print("Injecting burst " + str(time_start_isot) + " with DM = " + str(DM) + ", width = " + str(width) + ", S/N = " + str(SNR))

            image_tesseract = sim.make_image_cube(PSFimg=PSF,snr=SNR*100,width=width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=DM,output_file=log_file)

            #report in injections file
            with open(inject_file,"a") as csvfile:
                wr = csv.writer(csvfile,delimiter=',')
                wr.writerow([time_start_isot,DM,width,SNR])
            csvfile.close()

            #send
            for i in range(nchans):#NUM_CHANNELS//AVERAGING_FACTOR):
                #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
                msg=TXclient.send_data(time_start_isot, image_tesseract[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.port)
                if args.verbose: print(msg)
                time.sleep(1)

            

if __name__ == '__main__':
    main()


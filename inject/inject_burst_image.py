import numpy as np
from nsfrb import searching as sl
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





def make_image_cube(PSFimg=sl.default_PSF,snr=1000,width=5,loc=0.5,gridsize=sl.gridsize,nchans=sl.nchans,nsamps=sl.nsamps,RFI=False,DM=0,output_file="",datagridsize=sl.datagridsize):
    #get pngs
    """
    This function makes test images with finite width using Nikita's test pngs
    """



    dirname = cwd + "/simulations_and_classifications/src_examples/observation_2/images/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/simulations_and_classifications/src_examples/observation_2/images/"#testimgs_2024-03-18/"#{a}x{a}_images/"#src_examples/observation_1/images/".format(a=gridsize)
    pngs = os.listdir(dirname)
    sourceimg = np.zeros((gridsize,gridsize,nsamps,nchans))
    freqs = []
    fs = []

    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #need to rescale by 2
    snr = snr/2

    for png in pngs:
        print(png,file=fout)
        if ".png" in png:
            #get frequency
            freq = float(png[png.index("avg_") + 4: png.index("avg_") + 11])
            freqs.append(freq)
            fs.append(png)

    #need to check that datagridsize/gridsize compatible
    if datagridsize > gridsize and datagridsize%gridsize != 0:
        diff = datagridsize%gridsize
        datagridsizecut = datagridsize - diff
    elif datagridsize > gridsize:
        diff = 0
        datagridsizecut = datagridsize


    #print(str(datagridsizecut) + " " + str(datagridsize) + " " + str(gridsize),file=fout) 
    if datagridsize > gridsize:
        print("Downsampling by factor " + str(datagridsizecut//gridsize) + "...",file=fout,end="")
    freqs_sorted = np.sort(freqs)
    fs_sorted = [x for x, _ in sorted(zip(fs, freqs))]
    #downsample and copy over time and frequency axes
    for i in range(nchans):
        for j in range(nsamps):

            #print(np.asarray(ImageOps.grayscale(Image.open(dirname + fs_sorted[i]))).shape)
            fullim = np.asarray(ImageOps.grayscale(Image.open(dirname + fs_sorted[i])))
            print(fullim.shape,file=fout)

            if datagridsize == gridsize:
                sourceimg[:,:,j,i] = fullim

            elif datagridsize < gridsize:
                diff = gridsize - datagridsize
                fullim = np.pad(fullim, (diff//2,diff - (diff//2)),mode='constant')
                sourceimg[:,:,j,i] = fullim

            elif datagridsize > gridsize:
                if datagridsize%gridsize != 0:
                    fullim = fullim[diff//2:(diff//2) + datagridsizecut,diff//2:(diff//2)+ datagridsizecut]

                sourceimg[:,:,j,i] = fullim.reshape((gridsize,datagridsize//gridsize,gridsize,datagridsize//gridsize)).mean((1,3))


    #now add noise based on the SNR
    #PSFimg = make_PSF_cube(loc=loc,gridsize=gridsize,nchans=nchans,nsamps=nsamps)
    #sourceimg = sourceimg[gridsize//2:gridsize//2 + gridsize,gridsize//2:gridsize//2 + gridsize]
    noises = []
    for i in range(nchans):
        sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i] = sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]/(np.sum((PSFimg*sourceimg)[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]))#/np.sum(PSFimg[:,:,:,i]))


        print(np.sum((PSFimg*sourceimg)[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]),np.sum(PSFimg[:,:,:,i]),file=fout)


        #img[16,16,500:500+wid,:] = snr/wid
        sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i] = sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]*snr#/np.sum(PSFimg[:,:,0,i])

        sourceimg[:,:,:int(loc*nsamps),:] = 0
        sourceimg[:,:,int(loc*nsamps) + width:,:] = 0

    #if DM is given, disperse before adding noise
    if DM != 0:
        tmp,sourceimg = dedisperse_allDM(sourceimg,DM=-DM)[:,:,:,:,0]
    for i in range(nchans):
        sourceimg[:,:,:,i] += norm.rvs(loc=0,scale=np.sqrt(1/np.nansum(PSFimg[:,:,0,i])/width/nchans),size=(gridsize,gridsize,nsamps))
        noises.append(1/np.nansum(PSFimg[:,:,0,i])/width/nchans)

    if output_file != "":
        fout.close()
    return sourceimg


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
    PSF = sl.make_PSF_cube(gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,output_file=log_file)

    if args.nbursts == 1:
        #get current time
        time_start_isot = Time.now().isot
        print("Injecting burst " + str(time_start_isot) + " with DM = " + str(args.DM) + ", width = " + str(args.width) + ", S/N = " + str(args.SNR))
        
        image_tesseract = make_image_cube(PSFimg=PSF,snr=args.SNR,width=args.width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=args.DM,output_file=log_file)

        #send
        for i in range(nchans):#NUM_CHANNELS//AVERAGING_FACTOR):
            #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
            msg=TXclient.send_data(time_start_isot, image_tesseract[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.port)
            if args.verbose: print(msg)
            time.sleep(1)

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

            image_tesseract = make_image_cube(PSFimg=PSF,snr=SNR,width=width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=DM,output_file=log_file)


            #send
            for i in range(nchans):#NUM_CHANNELS//AVERAGING_FACTOR):
                #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
                msg=TXclient.send_data(time_start_isot, image_tesseract[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.port)
                if args.verbose: print(msg)
                time.sleep(1)

if __name__ == '__main__':
    main()


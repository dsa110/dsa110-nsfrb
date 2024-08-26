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
from nsfrb.imaging import uv_to_pix

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
cwd = os.environ['NSFRBDIR']
sys.path.append(cwd + "/")
from nsfrb.outputlogging import printlog
from simulations_and_classifications import generate_PSF_images as scPSF
from nsfrb.config import *
error_file = cwd + "-logfiles/inject_error_log.txt"
log_file = cwd + "-logfiles/inject_log.txt"
inject_file = cwd + "-injections/injections.csv"
cand_dir = cwd + "-candidates/"
psf_dir = cwd + "-PSF/"


def generate_inject_image(DEC=0,offsetRA=0,offsetDEC=0,snr=1000,width=5,loc=0.5,gridsize=gridsize,nchans=nchans,nsamps=nsamps,DM=0,output_file=log_file):
    """
    Uses functions from simulations_and_classifications to make injections
    """
    offsetRA = 0
    offsetDEC = 75
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    
    #for proper normalization need to scale snr
    snr = snr#*100/3
    
    
    #create a noiseless image
    PSFimg = scPSF.generate_PSF_images(psf_dir,DEC,gridsize//2,True,nsamps,dtype=np.float64)
    PSFimg -= np.nanmin(PSFimg)
    print("PSF shape:" + str(PSFimg.shape),file=fout)
    sourceimg=copy.deepcopy(PSFimg)

    #shift based on offsets
    if offsetRA > 0:
        sourceimg = np.pad(sourceimg,((0,0),(offsetRA,0),(0,0),(0,0)))[:,:gridsize,:,:]
    elif offsetRA < 0:
        sourceimg = np.pad(sourceimg,((0,0),(0,-offsetRA),(0,0),(0,0)))[:,-gridsize:,:,:]
    if offsetDEC > 0:
        sourceimg = np.pad(sourceimg,((offsetDEC,0),(0,0),(0,0),(0,0)))[:gridsize,:,:,:]
    elif offsetDEC < 0:
        sourceimg = np.pad(sourceimg,((0,-offsetDEC),(0,0),(0,0),(0,0)))[-gridsize:,:,:,:]
    print("IMG shape:"+str(sourceimg.shape),file=fout)
    #normalize based on snr
    for i in range(nchans):
        sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i] = sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]/(np.sum((PSFimg*sourceimg)[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]))#/np.sum(PSFimg[:,:,:,i]))


        print(np.sum((PSFimg*sourceimg)[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]),np.sum(PSFimg[:,:,:,i]),file=fout)


        #img[16,16,500:500+wid,:] = snr/wid
        sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i] = sourceimg[:,:,int(loc*nsamps) : int(loc*nsamps) + width,i]*snr#/np.sum(PSFimg[:,:,0,i])

        sourceimg[:,:,:int(loc*nsamps),:] = 0
        sourceimg[:,:,int(loc*nsamps) + width:,:] = 0


    #if DM is given, disperse before adding noise
    if DM != 0:
        sourceimg_dm = np.zeros(sourceimg.shape)
        freq_axis = np.linspace(fmin,fmax,nchans)
        for i in range(gridsize):
            for j in range(gridsize):
                tdelays = DM*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
                tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
                tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
                tdelays_frac = tdelays/tsamp - tdelays_idx_low

                for k in range(nchans):
                    #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac)
                    arrlow =  np.pad(sourceimg[i,j,:,k],((0,tdelays_idx_low[k])),mode="constant",constant_values=0)[tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
                    arrhi =  np.pad(sourceimg[i,j,:,k],((0,tdelays_idx_hi[k])),mode="constant",constant_values=0)[tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

                    sourceimg_dm[i,j,:,k] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])

 

    else:
        sourceimg_dm = sourceimg
    for i in range(nchans):
        sourceimg_dm[:,:,:,i] += norm.rvs(loc=0,scale=np.sqrt(1/np.nansum(PSFimg[:,:,0,i])/width/nchans),size=(gridsize,gridsize,nsamps))
    #    noises.append(1/np.nansum(PSFimg[:,:,0,i])/width/nchans)

    if output_file != "":
        fout.close()
    return sourceimg_dm



def main(args):
    #redirect stderr
    sys.stderr = open(error_file,"w")


    #make image
    #PSF = sim.make_PSF_cube(gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,output_file=log_file)

    if args.nbursts == 1:
        #get current time
        time_start = Time.now()
        time_start_isot = time_start.isot
        RA_axis,DEC_axis = uv_to_pix(time_start.mjd,args.gridsize,Lat=37.23,Lon=-118.2851)
        printlog("Injecting burst " + str(time_start_isot) + " with DM = " + str(args.DM) + ", width = " + str(args.width) + ", S/N = " + str(args.SNR),output_file=log_file)
        printlog("RA=" + str(np.nanmean(RA_axis)),output_file=log_file)
        printlog("DEC="+str(np.nanmean(DEC_axis)),output_file=log_file)
        image_tesseract = generate_inject_image(DEC=np.nanmean(DEC_axis),offsetRA=np.random.choice(np.arange(-gridsize//3,gridsize//3,dtype=int)),offsetDEC=np.random.choice(np.arange(-gridsize//3,gridsize//3,dtype=int)),snr=args.SNR,width=args.width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=args.DM,output_file=log_file)
        #image_tesseract = sim.make_image_cube(PSFimg=PSF,snr=args.SNR*100,width=args.width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=args.DM,output_file=log_file)

        #send
        for i in range(args.nchansend):#NUM_CHANNELS//AVERAGING_FACTOR):
            #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
            msg=TXclient.send_data(time_start_isot, image_tesseract[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=60,port=args.port)
            if args.verbose: print(msg)
            time.sleep(1)

        #report in injections file
        with open(inject_file,"a") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            wr.writerow([time_start_isot,args.DM,args.width,args.SNR])
        csvfile.close()

    else:
        SNRs = norm.rvs(loc=args.SNR,scale=1,size=args.nbursts)

        if args.randomize:
            DMtrials = np.load(cand_dir + "DMtrials.npy")
            widthtrials = np.load(cand_dir + "widthtrials.npy")
            DMs = np.random.choice(DMtrials,args.nsamps,replace=True)
            widths = np.array(np.random.choice(widthtrials,args.nsamps,replace=True),dtype=int)
        else:
            widths = np.array(np.clip(norm.rvs(loc=args.width,scale=1,size=args.nbursts),0,args.nsamps),dtype=int)
            DMs = norm.rvs(loc=args.DM,scale=args.DM/10,size=args.nbursts)

        RA_axis,DEC_axis = uv_to_pix(time_start_isot,args.gridsize,Lat=37.23,Lon=-118.2851)
        RA = np.nanmean(RA_axis)
        DEC=np.nanmean(DEC_axis)
        offsetRAs = np.random.choice(np.arange(-gridsize//3,gridsize//3,dtype=int))
        offsetDECs = np.random.choice(np.arange(-gridsize//3,gridsize//3,dtype=int))

        for j in range(args.nbursts):
            SNR = SNRs[j]
            width = widths[j]
            DM = DMs[j] 

            #get current time
            time_start = Time.now()
            time_start_isot = time_start.isot
            RA_axis,DEC_axis = uv_to_pix(time_start.mjd,args.gridsize,Lat=37.23,Lon=-118.2851)
            RA = np.nanmean(RA_axis)
            DEC=np.nanmean(DEC_axis)

            printlog("Injecting burst " + str(time_start_isot) + " with DM = " + str(DM) + ", width = " + str(width) + ", S/N = " + str(SNR),output_file=log_file)
            printlog("RA=" + str(RA),output_file=log_file)
            printlog("DEC="+str(DEC),output_file=log_file)
            
            image_tesseract = generate_inject_image(DEC=DEC,offsetRA=offsetRAs[j],offsetDEC=offsetDECs[j],snr=SNR,width=width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=DM,output_file=log_file)
            #image_tesseract = sim.make_image_cube(PSFimg=PSF,snr=SNR*100,width=width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=DM,output_file=log_file)

            #report in injections file
            with open(inject_file,"a") as csvfile:
                wr = csv.writer(csvfile,delimiter=',')
                wr.writerow([time_start_isot,DM,width,SNR])
            csvfile.close()

            #send
            for i in range(args.nchansend):#NUM_CHANNELS//AVERAGING_FACTOR):
                #dirty_images_all_bytes = dirty_images_all.transpose((2, 3, 0, 1))[:,:,:,i].tobytes()
                msg=TXclient.send_data(time_start_isot, image_tesseract[:,:,:,i] ,verbose=args.verbose,retries=5,keepalive_time=10,port=args.port)
                if args.verbose: print(msg)
                time.sleep(1)

            #wait before next
            time.sleep(args.delay)

if __name__ == '__main__':
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
    parser.add_argument('--delay',type=float,help='If multiple bursts injected, the time in seconds between each burst; default 30 seconds',default=30)
    parser.add_argument('--randomize',action='store_true',default=False,help='randomize DM and widths over the search range')
    parser.add_argument('--nchansend',type=int,help='Number of channels to send, used for testing, default=16',default=16)
    args = parser.parse_args()
    main(args)


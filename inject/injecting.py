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
import glob
#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
#cwd = os.environ['NSFRBDIR']
#sys.path.append(cwd + "/")
from nsfrb.outputlogging import printlog
from simulations_and_classifications import generate_PSF_images as scPSF
from nsfrb.config import *
from nsfrb.searching import gen_dm,gen_dm_shifts,minDM,maxDM

"""
minDM = 171
maxDM = 4000
DM_trials = np.array(gen_dm(minDM,maxDM,1.5,fc*1e-3,nchans,tsamp,chanbw))#[0:1]
nDMtrials = len(DM_trials)
"""
#freq_axis = np.linspace(fmin,fmax,nchans)
#corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append,wraps_append,wraps_no_append = gen_dm_shifts(DM_trials,freq_axis,tsamp,nsamps,outputwraps=True)
"""
error_file = cwd + "-logfiles/inject_error_log.txt"
log_file = cwd + "-logfiles/inject_log.txt"
inject_file = cwd + "-injections/injections.csv"
cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/"#cwd + "-candidates/"
psf_dir = cwd + "-PSF/"
frame_dir = cwd + "-frames/"
noise_dir = cwd + "-noise/"
inject_dir = cwd + "-injections/"
"""
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,inject_log_file,chanbw,fc,fmin,fmax,bmin,freq_axis

PSFSUM = (3900/16) #(((20/300)**2)*3900/16)*np.sqrt(40/150)#*(300**2)



def generate_inject_image(isot,HA=0,DEC=0,offsetRA=0,offsetDEC=0,snr=1000,width=5,loc=0.5,gridsize=gridsize,nchans=nchans,nsamps=nsamps,DM=0,output_file=inject_log_file,maxshift=0,offline=False,noiseless=False,spacefilter=True,HA_axis=None,DEC_axis=None,noiseonly=True,bmin=bmin,robust=2):
    """
    Uses functions from simulations_and_classifications to make injections
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    
    #for proper normalization need to scale snr
    #snr = snr*100*1000*0.75*75/15#40.625#*100/3
    snr = snr*100*(1000/20)*16*((10/28)**2)#*1000/2 
    
   
    #estimate noise to inject from raw data
    #if not noiseless:
    #imgnoise = np.nanmean(np.load(noise_dir + "raw_noise_300x300.npy"))
    visnoise = np.nanmean(np.load(noise_dir + "raw_vis_noise_real.npy"))
    scalenoise = vis_to_img_slope #imgnoise/visnoise
        
    injectnoise = PSFSUM/((snr/width)*scalenoise)
    if not noiseless:
        print("INJECTING NOISE:" + str(injectnoise),file=fout)
        print("RESCALEFACTOR:" + str(scalenoise),file=fout)
    vi_scale = visnoise/injectnoise
    print("VISCALE:",vi_scale,file=fout)
    if noiseless:
        injectnoise = 0


    if HA_axis is not None and DEC_axis is not None:
        print("POINTING HA,DEC:",HA,DEC,file=fout)
        print("AXIS CENTER HA,DEC:",HA_axis[int(len(HA_axis)//2)],DEC_axis[int(len(DEC_axis)//2)],file=fout)
        print("SOURCE HA DEC:",HA_axis[int(len(HA_axis)//2) + offsetRA],DEC_axis[int(len(DEC_axis)//2) + offsetDEC],file=fout)

    #create a noiseless image
    os.system("mkdir " + inject_dir + "dataset_" + isot +"/")
    dataset_dir = inject_dir + "dataset_" + isot +"/"
    if not noiseonly:
        PSFimg = np.abs(scPSF.generate_PSF_images(dataset_dir,DEC*np.pi/180,gridsize,True,nsamps=width,dtype=np.float64,HA=HA*np.pi/180,injectnoise=injectnoise,
                                    srcHAoffset=0 if HA_axis is None else (HA_axis[int(len(HA_axis)//2) + offsetRA]-HA)*np.pi/180,
                                    srcDECoffset=0 if DEC_axis is None else (DEC_axis[int(len(DEC_axis)//2) + offsetDEC]-DEC)*np.pi/180,
                                    bmin=bmin,robust=robust))
        if width == 1:
            PSFimg = PSFimg[:,:,np.newaxis,:]


    if not noiseless:
        if not noiseonly: PSFimg *= vi_scale
        if offline:
            nn = nsamps+maxshift+maxshift
            if not noiseonly: nn -=width
            noiseimg = scPSF.generate_PSF_images(psf_dir,DEC*np.pi/180,gridsize,True,nn,dtype=np.float64,HA=HA*np.pi/180,injectnoise=injectnoise,noise_only=True,bmin=bmin,robust=robust)*visnoise/injectnoise
            if nsamps-width+maxshift+maxshift == 1:
                noiseimg = noiseimg[:,:,np.newaxis,:]
            last_frame = noiseimg[:,:,:maxshift,:]
            noiseimg = noiseimg[:,:,maxshift:,:]
            print("OFFLINE CASE MAXSHIFT:",maxshift,file=fout)
            f = open(frame_dir + "last_frame.npy","wb")
            np.save(f,last_frame)
            f.close()
        else:
            nn = nsamps+maxshift
            if not noiseonly: nn -= width
            noiseimg = scPSF.generate_PSF_images(psf_dir,DEC*np.pi/180,gridsize,True,nn,dtype=np.float64,HA=HA*np.pi/180,injectnoise=injectnoise,noise_only=True,bmin=bmin,robust=robust)*visnoise/injectnoise
            if nsamps-width+maxshift == 1:
                noiseimg = noiseimg[:,:,np.newaxis,:]
        
        noiseimg1 = noiseimg[:,:,:int(loc*nsamps)+maxshift,:]
        noiseimg2 = noiseimg[:,:,int(loc*nsamps)+maxshift:,:]
        
        if noiseonly:
            PSFimg = noiseimg
        else:
            PSFimg = np.concatenate([noiseimg2,PSFimg,noiseimg1],axis=2)
        print(noiseimg1.shape,noiseimg2.shape,noiseimg.shape,PSFimg.shape,vi_scale,file=fout)
    else:
        nn = nsamps+maxshift
        if not noiseonly: nn -= width
        PSFimg = np.concatenate([np.zeros((gridsize,gridsize,(int(loc*nsamps)+maxshift),nchans)),PSFimg,np.zeros((gridsize,gridsize,nn-(int(loc*nsamps)+maxshift),nchans))],axis=2)[:,:,::-1,:]*vi_scale
        print(nn,PSFimg.shape,vi_scale,file=fout)

    print("PSF MEAN:" + str(np.nanmean(PSFimg)),file=fout)
    print("PSF MEDIAN:" + str(np.nanmedian(PSFimg)),file=fout)
    print("PSF MAX:" + str(np.nanmax(PSFimg)),file=fout)
    print("PSF SUM:" + str(np.nansum(PSFimg[:,:,0,:])),file=fout)
    print("PSF SUM/CHANNEL:" +  str(np.nansum(PSFimg[:,:,0,:],axis=(0,1))),file=fout)
    #PSFimg -= np.nanmin(PSFimg)
    print("PSF MIN: " + str(np.nanmin(PSFimg)),file=fout)
    print("PSF shape:" + str(PSFimg.shape),file=fout)
    sourceimg=copy.deepcopy(PSFimg)


    print("IMG shape:"+str(sourceimg.shape),file=fout)
    

    #if DM is given, disperse before adding noise
    if DM != 0:
        print("COMPUTING SHIFTS FOR DM=",DM,"pc/cc",file=fout)
        DM_trials = np.array(gen_dm(minDM,maxDM,1.5,fc*1e-3,nchans,tsamp,chanbw,nsamps))#[0:1]
        #freq_axis = np.linspace(fmin,fmax,nchans)
        corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append,wraps_append,wraps_no_append = gen_dm_shifts(np.array([DM]),freq_axis,tsamp,nsamps,outputwraps=True,maxshift=maxshift)

        #nsamps = sourceimg.shape[-2]
        DM_idx = 0#list(DM_trials).index(DM)
        print("PRE-DM SHAPE:",sourceimg.shape,file=fout)
        sourceimg_dm = (((((np.take_along_axis(sourceimg[:,:,::-1,np.newaxis,:].repeat(1,axis=3).repeat(2,axis=4),indices=corr_shifts_all_append[:,:,:,DM_idx:DM_idx+1,:],axis=2))*tdelays_frac_append[:,:,:,DM_idx:DM_idx+1,:]))[:,:,:,0,:]))
        print("POST-DM SHAPE:",sourceimg_dm.shape,file=fout)
        #zero out anywhere that was wrapped
        #sourceimg_dm[wraps_no_append[:,:,:,DM_idx,:].repeat(sourceimg.shape[0],axis=0).repeat(sourceimg.shape[1],axis=1)] = 0

        #now average the low and high shifts 
        sourceimg_dm = (sourceimg_dm.reshape(tuple(list(sourceimg.shape)[:2] + [nsamps,nchans] + [2])).sum(4))[:,:,::-1,:]
    else:
        sourceimg_dm = sourceimg

    np.save(inject_dir + "testimg",sourceimg_dm)
    sourceimg_dm = sourceimg_dm[:,:,:nsamps,:]
    print("FINAL IMG SHAPE:" + str(sourceimg_dm.shape),file=fout)


    if output_file != "":
        fout.close()
    return sourceimg_dm

"""
default_DMtrials = np.load(cand_dir + "DMtrials.npy")
default_widthtrials = np.load(cand_dir + "widthtrials.npy")
#freq_axis = np.linspace(fmin,fmax,nchans)
tDM_max = (4.15)*np.max(default_DMtrials)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms
maxshift = int(np.ceil(tDM_max/tsamp))
"""
from nsfrb.searching import DM_trials as default_DMtrials
from nsfrb.searching import widthtrials as default_widthtrials
#from nsfrb.searching import maxshift
def draw_burst_params(time_start_isot,RA_axis=None,DEC_axis=None,DM=np.nan,width=np.nan,SNR=np.nan,gridsize=gridsize,DMtrials=default_DMtrials,widthtrials=default_widthtrials,freq_axis=freq_axis,nsamps=nsamps,nchans=nchans,tsamp=tsamp,SNRmin=0,SNRmax=10000,output_file=inject_log_file):
    """
    Randomly draws injected burst parameters from set of trial DMs, widths and RA/DEC grid
    """
    print("FROM DRAW PARAMS:",tsamp)
    #DMtrials = np.array(gen_dm(minDM,maxDM,1.5,fc*1e-3,nchans,tsamp,chanbw,nsamps))
    #DMtrials = DMtrials[DMtrials<2000]
    #widthtrials = widthtrials[widthtrials<int(nsamps//2)]
    
    #get RA,DEC axes
    time_start = Time(time_start_isot,format='isot')
    if RA_axis is None or DEC_axis is None:
        RA_axis,DEC_axis,tmp = uv_to_pix(time_start.mjd,gridsize,Lat=Lat,Lon=Lon)
    offsetRA = np.random.choice(np.arange(-gridsize//3,gridsize//3,dtype=int))
    offsetDEC = np.random.choice(np.arange(-gridsize//3,gridsize//3,dtype=int))

    #draw random DM, width, SNR if not specified
    tDM_max = (4.15)*np.max(DMtrials)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms
    maxshift = int(np.ceil(tDM_max/tsamp))
    if np.isnan(DM):
        DM = np.random.choice(DMtrials[DMtrials<2000])
    if np.isnan(width):
        width = np.random.choice(widthtrials[widthtrials<int(nsamps//2)])
    if np.isnan(SNR):
        SNR = uniform.rvs(loc=SNRmin,scale=SNRmax)

    printlog("Injecting burst " + str(time_start_isot) + " with DM = " + str(DM) + ", width = " + str(width) + ", S/N = " + str(SNR),output_file=inject_log_file)
    printlog("RA=" + str(RA_axis[int(len(RA_axis)//2 + offsetRA)]),output_file=inject_log_file)
    printlog("DEC="+str(DEC_axis[int(len(DEC_axis)//2 + offsetDEC)]),output_file=inject_log_file)

    return offsetRA,offsetDEC,SNR,width,DM,maxshift

"""
    #generate burst
    image_tesseract = generate_inject_image(DEC=DEC_axis[int(len(DEC_axis)//2)],offsetRA=offsetRA,offsetDEC=offsetDEC,snr=SNR,width=width,loc=0.5,gridsize=gridsize,nchans=nchans,nsamps=nsamps,DM=DM,output_file=log_file,maxshift=maxshift,offline=False,noiseless=True)


def main(args):
    #redirect stderr
    sys.stderr = open(error_file,"w")


    #make image
    #PSF = sim.make_PSF_cube(gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,output_file=log_file)
    DMtrials = np.load(cand_dir + "DMtrials.npy")
    widthtrials = np.load(cand_dir + "widthtrials.npy")
    freq_axis = np.linspace(fmin,fmax,nchans)
    tDM_max = (4.15)*np.max(DMtrials)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms
    maxshift = int(np.ceil(tDM_max/tsamp))
    if args.nbursts == 1:
        #get current time
        time_start = Time.now()
        time_start_isot = time_start.isot
        RA_axis,DEC_axis = uv_to_pix(time_start.mjd,args.gridsize,Lat=37.23,Lon=-118.2851)
        printlog("Injecting burst " + str(time_start_isot) + " with DM = " + str(args.DM) + ", width = " + str(args.width) + ", S/N = " + str(args.SNR),output_file=log_file)
        printlog("RA=" + str(np.nanmean(RA_axis)),output_file=log_file)
        printlog("DEC="+str(np.nanmean(DEC_axis)),output_file=log_file)
        image_tesseract = generate_inject_image(DEC=np.nanmean(DEC_axis),offsetRA=np.random.choice(np.arange(-gridsize//3,gridsize//3,dtype=int)),offsetDEC=np.random.choice(np.arange(-gridsize//3,gridsize//3,dtype=int)),snr=args.SNR,width=args.width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=args.DM,output_file=log_file,maxshift=maxshift,offline=args.offline)
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
            
            image_tesseract = generate_inject_image(DEC=DEC,offsetRA=offsetRAs[j],offsetDEC=offsetDECs[j],snr=SNR,width=width,loc=0.5,gridsize=args.gridsize,nchans=args.nchans,nsamps=args.nsamps,DM=DM,output_file=log_file,maxshift=maxshift,offline=args.offline)
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
    parser.add_argument('--offline',action='store_true',default=False,help='Run offline injection system')
    args = parser.parse_args()
    main(args)
"""

import numpy as np
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

from scipy.interpolate import interp1d
from scipy.ndimage import convolve
from scipy.signal import convolve2d

from concurrent.futures import ProcessPoolExecutor, as_completed
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
f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()
sys.path.append(cwd + "/") 
from nsfrb.config import *


#output_dir = cwd + "/tmpoutput/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/"
coordfile = cwd + "/DSA110_Station_Coordinates.csv" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/DSA110_Station_Coordinates.csv"
output_file = cwd + "-logfiles/search_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/search_log.txt"
cand_dir = cwd + "/candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
f=open(output_file,"w")
f.close()



"""
Search parameters
"""
#tsamp = 130 #ms
#T = 3250 #ms
nsamps = int(T/tsamp)

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

#create axes
RA_axis = np.linspace(RA_point-pixsize*gridsize//2,RA_point+pixsize*gridsize//2,gridsize)
DEC_axis = np.linspace(DEC_point-pixsize*gridsize//2,DEC_point+pixsize*gridsize//2,gridsize)
time_axis = np.linspace(0,T,nsamps) #ms
freq_axis = np.linspace(fmin,fmax,nchans) #MHz

#width trials
nwidths=1#4
widthtrials =np.array([1,2]) #np.logspace(0,3,nwidths,base=2,dtype=int)

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
minDM = 171
maxDM = 4000
DM_trials = np.array(gen_dm(minDM,maxDM,1.5,fc*1e-3,nchans,tsamp,chanbw))#[0:1]
DM_trials = np.concatenate([[0],DM_trials])
DM_trials = np.array([0,100])
nDMtrials = len(DM_trials)

#snr threshold
SNRthresh = 6


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

"""Search functions"""
datagridsize = 256
def make_PSF_cube(gridsize=gridsize,nchans=nchans,nsamps=nsamps,RFI=False,output_file=output_file):
    """
    This function creates a frequency-dependent PSF based on Nikita's source simulation pipeline. It
    uses pre-defined images and downsamples to the desired resolution. The PSF is duplicated along the time
    axis.   
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    #get pngs for a point source from Nikita's images
    dirname = cwd + "/simulations_and_classifications/src_examples/observation_2/images/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/simulations_and_classifications/src_examples/observation_2/images/"
    pngs = os.listdir(dirname)
    sourceimg = np.zeros((gridsize,gridsize,nsamps,nchans))
    freqs = []
    fs = []
	    
    print("Creating PSF with shape " + str(sourceimg.shape) + " using " + str(datagridsize) + "x" + str(datagridsize) + " images in " + dirname + "...",file=fout)
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

    #roll if not perfectly centered
    maxpix = tuple(np.array(np.unravel_index(np.argmax(sourceimg[:,:,0,0].flatten()),(gridsize,gridsize))))
    centerpix = ((gridsize//2) - 1,(gridsize//2) - 1)
    if maxpix != centerpix:
    
        rolledPSFimg = np.roll(np.roll(sourceimg,shift=centerpix[0]-maxpix[0],axis=0),shift=centerpix[1]-maxpix[1],axis=1)
        rolledPSFimg[gridsize - (maxpix[0]-centerpix[0]):,:,:,:] = 0
        rolledPSFimg[:,gridsize - (maxpix[1]-centerpix[1]):,:,:] = 0
    else: rolledPSFimg = PSFimg
    #cutout image
    #PSFimg = rolledPSFimg[gridsize//2:gridsize//2 + gridsize,gridsize//2:gridsize//2 + gridsize]

    print("Complete!",file=fout)
    if output_file != "":
        fout.close()
    return rolledPSFimg

default_PSF = make_PSF_cube()


def make_image_cube(PSFimg=default_PSF,snr=1000,width=5,loc=0.5,gridsize=gridsize,nchans=nchans,nsamps=nsamps,RFI=False,DM=0,output_file=""):
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
        tmp,sourceimg = dedisperse(sourceimg,DM=-DM)
    for i in range(nchans):
        sourceimg[:,:,:,i] += norm.rvs(loc=0,scale=np.sqrt(1/np.nansum(PSFimg[:,:,0,i])/width/nchans),size=(gridsize,gridsize,nsamps))
        noises.append(1/np.nansum(PSFimg[:,:,0,i])/width/nchans)

    if output_file != "":
        fout.close()
    return sourceimg




from scipy.signal import convolve2d
from scipy.signal import correlate2d
def matched_filter_space(image_tesseract,PSFimg,usefft=False):
    """
    Matched filter via convolution w/ DSA-110 core PSF
    """

    image_tesseract_filtered = np.zeros(image_tesseract.shape)
    nsamps = image_tesseract.shape[2]
    nchans = image_tesseract.shape[3]
    for i in range(nsamps):
        for j in range(nchans):
            if usefft:
                image_tesseract_filtered[:,:,i,j] =  np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(image_tesseract[:,:,i,j])*np.fft.fft2(PSFimg[:,:,i,j]))))#np.abs(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(np.fft.fft(image_tesseract[:,:,i,j]))*np.fft.fftshift(np.fft.fft(PSFimg[:,:,i,j])))))
            else:
                image_tesseract_filtered[:,:,i,j] = convolve2d(image_tesseract[:,:,i,j],PSFimg[:,:,i,j],mode='same') #assume the PSF is already centered

    return image_tesseract_filtered
    
    
    #np.nansum(np.nansum((img/np.array(noises)),3)*np.nanmean(PSFimg,3)/(np.nansum(1/np.array(noises))),axis=(0,1))


def snr_vs_RA_DEC_new(image_tesseract_filtered_dm,wid,mode='4d',noiseth=1/10,plot=False,output_file=""):
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
    for i in range(gridsize_DEC):
        for j in range(gridsize_RA):
            timeseries = image_tesseract_filtered_dm[i,j,:]
            csig = np.convolve(np.nan_to_num(timeseries,nan=0),boxcar,'same')#/wid#/np.sum(boxcar)
            peakidx = np.argmax(csig)


            #print(np.argmin(csig-np.max(csig)/2),nsamps-np.argmin(csig[::-1]-np.max(csig)/2))
            
            
            s=np.nanstd(csig[csig<noiseth*np.nanmax(csig)])#np.nanstd(np.concatenate([csig[:np.nanargmax(csig)-wid],csig[np.nanargmax(csig)+wid+1:]]))
            noisemap[i,j] = s

            

            #print(np.nanmax(csig),s,np.nanmax(csig)/s)
            if s == 0: image_tesseract_binned[i,j] = np.nan
            else: image_tesseract_binned[i,j] = np.nanmax(csig)/s#/wid
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



# Brute force dedispersion
def dedisperse(image_tesseract_point,DM,tsamp=tsamp,freq_axis=freq_axis,output_file=""):
    """
    This function dedisperses a dynamic spectrum pixel grid of shape gridsize x gridsize x nsamps x nchans by brute force without accounting for edge effects
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

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
    print("Done!",file=fout)
    if output_file != "":
        fout.close()
    return dedisp_timeseries_all,dedisp_img


def dedisperse_1D(image_tesseract_point,DM,tsamp=tsamp,freq_axis=freq_axis,output_file=""):
    """
    This function dedisperses a dynamic spectrum of shape nsamps x nchans by brute force without accounting for edge effects
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    #get delay axis
    tdelays = DM*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
    tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
    tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
    tdelays_frac = tdelays/tsamp - tdelays_idx_low
    print("Trial DM: " + str(DM) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",file=fout,end="")
    nchans = len(freq_axis)
    dedisp_timeseries = np.zeros(image_tesseract_point.shape[0])
    #shift each channel
    for k in range(nchans):
        #print(tdelays_idx_hi,tdelays_idx_low,tdelays_frac)
        arrlow =  np.pad(image_tesseract_point[:,k],((0,tdelays_idx_low[k])),mode="constant",constant_values=0)[tdelays_idx_low[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)
        arrhi =  np.pad(image_tesseract_point[:,k],((0,tdelays_idx_hi[k])),mode="constant",constant_values=0)[tdelays_idx_hi[k]:]/nchans#np.roll(image_tesseract_intrinsic[:,:,:,k],tdelays_idx[k],axis=2)

        dedisp_timeseries += arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])
    print("Done!",file=fout)
    if output_file != "":
        fout.close()
    return dedisp_timeseries


#Updated search code
#run search pipeline with desired DM, width trial range; output candidates to a csv? pkl? txt?
#takes 4D cube (RA,DEC,TIME,FREQUENCY)

def run_search_new(image_tesseract,RA_axis=RA_axis,DEC_axis=DEC_axis,time_axis=time_axis,freq_axis=freq_axis,
                   DM_trials=DM_trials,widthtrials=widthtrials,tsamp=tsamp,SNRthresh=SNRthresh,plot=False,
                   off=10,PSF=default_PSF,offpnoise=0.3,verbose=False,output_file="",noiseth=1e-2,canddict=dict(),usefft=False,
                   multithreading=False,nrows=1,ncols=1,space_filter=True,raidx_offset=0,decidx_offset=0,dm_offset=0,threadDM=False):

    """
    This function takes an image cube of shape npixels x npixels x nchannels x ntimes and runs a dedispersion search that returns
    a list of candidates' DM, pulse width, RA, declination, and time of arrival(?)
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout

    printprefix = "[" + str(raidx_offset) + str(decidx_offset) + str(dm_offset) + "]" 


    #get axis sizes
    gridsize_RA = len(RA_axis)
    gridsize_DEC = len(DEC_axis)
    gridsize = gridsize_RA
    nsamps = len(time_axis)
    nchans = len(freq_axis)

    if space_filter:
        assert(gridsize_RA == gridsize_DEC)
        #create PSF if the shape doesn't match
        if PSF.shape != image_tesseract.shape:
            print(printprefix + "Updating PSF...",file=fout)
            PSF = make_PSF_cube(gridsize=gridsize,nsamps=nsamps,nchans=nchans)

        #2D matched filter for each timestep and channel
        print(printprefix +"Spatial matched filtering with DSA PSF...",file=fout)
        if usefft:
            print(printprefix +"Using 2D FFT method...",file=fout)
        image_tesseract_filtered = matched_filter_space(image_tesseract,PSF,usefft=usefft)
        print(printprefix +"Done!",file=fout)
        print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_filtered))),file=fout)
    else: 
        image_tesseract_filtered = image_tesseract
    

    #use the concurrent futures package to search sub-images separately; we have to do this AFTER the spatial matched filter so that the PSF structure is suppressed
    if multithreading: 
        #initialize a pool of processes for concurent execution
        if threadDM:maxProcesses = nrows*ncols*len(DM_trials)
        else: maxProcesses = nrows*ncols
        executor = ProcessPoolExecutor(maxProcesses) 

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
                                    False,1,1,False,col*gridsize_RA_i,row*gridsize_DEC_i,k,False))
                
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
                                    False,1,1,False,col*gridsize_RA_i,row*gridsize_DEC_i,0,False))


        for future in as_completed(task_list):
            print("---> Result " + str(i) + ":",file=fout)
            candidxs_i,cands_i,image_tesseract_binned_i,image_tesseract_filtered_i,canddict_i,DM_trials_i = future.result()
            if threadDM: subDMidx = list(DM_trials).index(DM_trials_i[0])#np.argmin(np.abs(DM_trials_i[0] - DM_trials))


            #save the binned image and candidates
            candidxs = list(candidxs) + list(candidxs_i)
            cands = list(cands) + list(cands_i)
            if threadDM: image_tesseract_binned[row*gridsize_DEC_i:(row+1)*gridsize_DEC_i,col*gridsize_RA_i:(col+1)*gridsize_RA_i,:,subDMidx:subDMidx+1] = image_tesseract_binned_i    
            else: image_tesseract_binned[row*gridsize_DEC_i:(row+1)*gridsize_DEC_i,col*gridsize_RA_i:(col+1)*gridsize_RA_i,:,:] = image_tesseract_binned_i

            for k in canddict_i.keys():
                canddict[k] = np.concatenate([canddict[k],canddict_i[k]])

        #make a dictionary for easy plotting of results
        ncands = len(cands)
    else: #proceed normally


        #dedisperse --> gridsize x gridsize x time x DM
        nDMtrials = len(DM_trials)
        print(printprefix +"Starting dedispersion with " + str(nDMtrials) + " trials...",file=fout)
        image_tesseract_dedisp = np.zeros((gridsize_DEC,gridsize_RA,nsamps,nDMtrials)) #stores output array as dedispersion transform for every pixel
        for d in range(nDMtrials):
            image_tesseract_dedisp[:,:,:,d] = dedisperse(image_tesseract_filtered,DM=DM_trials[d],tsamp=tsamp,freq_axis=freq_axis,output_file=output_file)[0]
        #print(image_tesseract_dedisp.shape)
        print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_dedisp))),file=fout)

        print(printprefix +"Done!",file=fout) 


        #boxcar filter and get snr using rolled PSF --> gridsize x gridsize x width x DM (x TOA?)
        nwidthtrials = len(widthtrials)
        image_tesseract_binned = np.zeros((gridsize_DEC,gridsize_RA,nwidthtrials,nDMtrials)) #stores output array as S/N for each dedispersion and width trial for every pixel
    
        print(printprefix +"Starting boxcar filtering with " + str(nwidthtrials) + " trials...",file=fout)
        #PSF parameters
        maxs = []
        maxs2 = []
        for w in range(nwidthtrials):
            for d in range(nDMtrials):
                image_tesseract_binned[:,:,w,d] = snr_vs_RA_DEC_new(image_tesseract_dedisp[:,:,:,d],widthtrials[w],noiseth=noiseth,output_file=output_file) 
                if d ==0 and plot:
                    maxs.append(image_tesseract_binned[15, 16,w,d])
                elif plot:
                    maxs2.append(image_tesseract_binned[15, 16,w,d])
        print(printprefix +"Done!",file=fout)    
        print(printprefix +"---> " + str(np.sum(np.isnan(image_tesseract_binned))),file=fout)

        if plot:
            plt.figure(figsize=(12,6))
            plt.plot(widthtrials,maxs,'o-')
            #plt.plot(widthtrials,maxs2,'o-')
            #plt.plot(np.arange(1,10),maxs[np.argmin(np.abs(widthtrials-5))]*np.sqrt(5/np.arange(1,10)),color='red')
            #plt.plot(np.arange(1,10),maxs[np.argmin(np.abs(widthtrials-5))]*np.sqrt(np.arange(1,10)/5),color='blue')
            plt.show()
        


        print(printprefix +"Searching for candidates with S/N > " + str(SNRthresh) + "...",file=fout)
        #find candidates above SNR threshold
        condition = (image_tesseract_binned>SNRthresh).flatten()
        ncands = np.sum(condition)
        canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs=np.unravel_index(np.arange(gridsize_DEC*gridsize_RA*nDMtrials*nwidthtrials)[condition],(gridsize_DEC,gridsize_RA,nwidthtrials,nDMtrials))#[1].shape
    
        canddecs = DEC_axis[canddec_idxs]
        candras = RA_axis[candra_idxs]
        candwids = widthtrials[candwid_idxs]
        canddms = DM_trials[canddm_idxs]
        candsnrs = image_tesseract_binned.flatten()[condition]
    
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

    print(printprefix +"Done! Found " + str(ncands) + " candidates",file=fout)
    if output_file != "":
        fout.close()
    return candidxs,cands,image_tesseract_binned,image_tesseract_filtered,canddict,DM_trials




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

        """
        plt.figure(figsize=(24,24))
        ax = plt.subplot(projection="3d")
        for k in classnames:
            if k != -1:
                c=ax.scatter(raidxs[classes==k],decidxs[classes==k],dmidxs[classes==k],s=100*(2**widthidxs[classes==k]),marker='o',alpha=0.1)
                ax.scatter(centroid_ras[k],centroid_decs[k],centroid_dms[k],s=100*(2**centroid_widths[k]),marker='v',color=c.get_facecolor(),alpha=1)
        if noisepoints > 0:
            c=ax.scatter(raidxs[classes==-1],decidxs[classes==-1],dmidxs[classes==-1],s=100*(2**widthidxs[classes==-1]),marker='o',alpha=0.1,color='grey')

        plt.savefig(output_dir + "hdbscan_cluster_plot.png")
        if show:	
            plt.show()
        else:
            plt.close()

        """
    return classes,centroid_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs

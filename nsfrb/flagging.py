import numpy as np
from dsacalib.utils import Direction
from dsautils.coordinates import create_WCS,get_declination,get_elevation
from nsfrb.outputlogging import printlog
from scipy.interpolate import interp1d
from astropy import wcs
from scipy.fftpack import ifftshift, ifft2,fftshift,fft2,fftfreq
from nsfrb.config import IMAGE_SIZE,UVMAX,flagged_antennas,crpix_dict,NUM_CHANNELS,AVERAGING_FACTOR
#modules for position and RA/DEC calibration
from influxdb import DataFrameClient
from astropy.coordinates import EarthLocation, AltAz, ICRS,SkyCoord,FK5
import astropy.units as u
from astropy.time import Time
import sys
from matplotlib import pyplot as plt
from nsfrb import simulating#,planning
import copy
import numba
import os
from scipy.optimize import curve_fit
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,Lon,Lat,az_offset,Height,flagged_antennas,flagged_corrs,tsamp


def flag_vis(dat, bname, blen, UVW, antenna_order, flagged_antennas, bmin=0, flagged_corrs=np.array([]),flag_channel_templates=[],realtime=False,sb=0,bmax=np.inf,flagged_chans=np.array([]),flagged_baseline_idxs=np.array([]),verbose=False,returnidxs=False):
    """
    Removes visibilities containing flagged antennas and below minimum 
    baseline length
    
    dat: visibility data (time x baseline x channel x pol)
    bname: baseline names
    blen: baseline lengths (baseline x 3)
    UVW: U,V,W, coordinates (baseline x 3)
    antenna_order: antenna ordering
    flagged_antennas: number of antennas to be flagged
    bmin: minimum baseline length in meters

    """
    flagged_vis = []
    for i in flagged_antennas:
        #print(i)
        for j in np.array(antenna_order):
            #print("--",j)
            if str(i) + "-" + str(j) in list(bname):
                flagged_vis.append(list(bname).index(str(i) + "-" + str(j)))
            elif str(j) + "-" + str(i) in list(bname):
                flagged_vis.append(list(bname).index(str(j) + "-" + str(i)))

        """
        for j in np.array(antenna_order)[:antenna_order.index(i)]:
            flagged_vis.append(list(bname).index(str(j) + "-" + str(i)))
        for j in np.array(antenna_order)[antenna_order.index(i):]:
            flagged_vis.append(list(bname).index(str(i) + "-" + str(j)))
        """
    for i in flagged_baseline_idxs:
        if list(bname.index.values).index(i) not in flagged_vis:
            flagged_vis.append(list(bname.index.values).index(i))

    flagged_vis = np.array(flagged_vis,dtype=int)
    if verbose: print("Flagged visibilities:",np.array(list(bname))[flagged_vis])
    unflagged_vis = np.array(list(set(np.arange(len(bname)))-set(flagged_vis)),dtype=int)
    if verbose: print("Unflagged visibilities:",np.array(list(bname))[unflagged_vis])
    antenna_order = list(set(antenna_order)-set(flagged_antennas))
    bname = bname[np.array(bname.index.values.tolist())[unflagged_vis]]
    blen = blen[unflagged_vis]
    UVW = UVW[:,unflagged_vis,:]
    dat = dat[:,unflagged_vis,:,:]

    #print(dat)
    #remove short baselines
    #if bmin > 0:
        
    blen_mask = np.logical_and(np.sqrt(UVW[0,:,0]**2 + UVW[0,:,1]**2)>=bmin,np.sqrt(UVW[0,:,0]**2 + UVW[0,:,1]**2)<bmax)
    bname = bname[np.array(bname.index.values.tolist())[blen_mask]]
    blen = blen[blen_mask]
    UVW = UVW[:,blen_mask,:]
    dat = dat[:,blen_mask,:,:]
    unflagged_vis = unflagged_vis[blen_mask]

    #flag corrs
    if len(flagged_corrs) > 0:
        if realtime:
            nchans_per_node = dat.shape[2]
            if sb in flagged_corrs:
                dat[:,:,:,:] = np.nan
        else:
            nchans_per_node = int(dat.shape[2]//int(NUM_CHANNELS//AVERAGING_FACTOR))
            for c in flagged_corrs:
                dat[:,:,c*nchans_per_node:(c+1)*nchans_per_node,:] = np.nan
    if len(flagged_chans) > 0:
        if realtime:
            nchans_per_node = dat.shape[2]
            dat[:,:,flagged_chans[np.logical_and(flagged_chans>=nchans_per_node*sb,flagged_chans<nchans_per_node*(sb+1))]-int(sb*nchans_per_node),:] = np.nan
        else:
            dat[:,:,flagged_chans,:] = np.nan
    if len(flag_channel_templates) > 0:
        for fct in flag_channel_templates:
            flag_channels = fct(dat)
            if len(flag_channels)>0:
                dat[:,:,flag_channels,:] = np.nan
            if verbose: print("Flagging channels:",flag_channels,"using template",fct)
    if returnidxs:
        return dat, bname, blen, UVW, antenna_order,unflagged_vis
    return dat, bname, blen, UVW, antenna_order

def fct_FRCBAND(dat):
    """
    Removes visibilities in 1435-1525 MHz military allocation (top 6 subbands)

    dat: visibility data (time x baseline x channel x pol)
    """
    nchans_per_node = int(dat.shape[2]//int(NUM_CHANNELS//AVERAGING_FACTOR))
    return ((int(NUM_CHANNELS//AVERAGING_FACTOR))*nchans_per_node) - np.arange(int(6*nchans_per_node)) - 1


def fct_BPASSBURST(dat,noise_dir=noise_dir,weights=[1,1]):
    """
    Removes visibilities in particular channel if surpassing 50x the accumulated 
    average in any timestep.

    dat: visibility data (time x baseline x channel x pol)
    """

    #compute new mean
    dat_test = np.nanmax(np.abs(np.nanmean(dat,axis=3)),axis=0) -  np.nanmedian(np.abs(np.nanmean(dat,axis=3)),axis=0) 
    dat_mean = np.nanmedian(np.nanmean(np.abs(dat_test),0))#np.nanmean(np.nanmedian(np.abs(np.nanmean(dat,axis=3)),0))

    #get the current running mean
    dat_run_mean = np.load(noise_dir+"running_vis_mean_burst.npy",allow_pickle=True)

    #create new mean
    if np.sum(dat_run_mean) is not None:
        dat_new_mean = (dat_mean*weights[0] + dat_run_mean*weights[1])/np.sum(weights)
    else:
        dat_new_mean = dat_mean
    np.save(noise_dir+"running_vis_mean_burst.npy",dat_new_mean)
    

    #compare to threshold
    flag_channels= np.arange(dat.shape[2])[np.nanmean(np.abs(dat_test),0)>10*dat_mean]
    return flag_channels


def fct_BPASS(dat,noise_dir=noise_dir,weights=[1,1]):
    """
    Removes visibilities in particular channel if surpassing 10x the accumulated 
    average.

    dat: visibility data (time x baseline x channel x pol)
    """

    #compute new mean
    dat_test = np.nanmean(dat,axis=(0,3))
    dat_mean = np.nanmedian(np.nanmean(np.abs(dat_test),0))

    #get the current running mean
    dat_run_mean = np.load(noise_dir+"running_vis_mean.npy",allow_pickle=True)

    #create new mean
    if np.sum(dat_run_mean) is not None:
        dat_new_mean = (dat_mean*weights[0] + dat_run_mean*weights[1])/np.sum(weights)
    else:
        dat_new_mean = dat_mean
    np.save(noise_dir+"running_vis_mean.npy",dat_new_mean)

    #compare to threshold
    flag_channels= np.arange(dat.shape[2])[np.nanmean(np.abs(dat_test),0)>5*dat_mean]
    return flag_channels

def fct_SWAVE(dat,RMS_THRESHOLD=1.0,STD_THRESHOLD=0.2,SLOPE_FIT=945.4546757820716):
    """
    Removes visibilities in particular channel based on specific conditions
    observed in past RFI. Conditions are specified as the function template()

    dat: visibility data (time x baseline x channel x pol)
    """
    nchans_per_node = int(dat.shape[2]//int(NUM_CHANNELS//AVERAGING_FACTOR))
    #print(nchans_per_node)
    flagged_channels = []
    for j in range(dat.shape[2]//nchans_per_node):
        #print(dat[:,:,nchans_per_node*j:nchans_per_node*(j+1),:])
        binned = np.nanmean(np.abs(np.nanmean(dat[:,:,nchans_per_node*j:nchans_per_node*(j+1),:],(2,3))),1)

                #weights=np.nanmax(np.abs(np.nanmean(dat[:,:,nchans_per_node*j:nchans_per_node*(j+1),:],(2,3))),0),axis=1)
        #print(binned)
        if ~np.all(np.isnan(binned)):
            popt,pcov = curve_fit(lambda x,b: b + SLOPE_FIT*x,np.arange(dat.shape[0])[~np.isnan(binned)]*tsamp/1000,binned[~np.isnan(binned)])
            tot_std = np.nanstd(binned)/np.nanmedian(np.abs(dat))
            tot_rms = np.sqrt(np.sum((binned - (popt[0] + SLOPE_FIT*np.arange(25)*tsamp/1000))**2))/np.nanmedian(np.abs(dat))#*weight
            print("Channel",j,",RMS:",tot_rms,",STD:",np.nanstd(binned))
            if tot_rms<RMS_THRESHOLD and tot_std>STD_THRESHOLD:
                print("-->FLAG")
                flagged_channels.append(j)
        else:
            print("Skipping Channel",j)
    flagged_channels = np.array(flagged_channels)
    return flagged_channels





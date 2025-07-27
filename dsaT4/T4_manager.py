from dsamfs import utils as pu
import glob
from astropy.coordinates import SkyCoord
from astropy import units as u
from nsfrb import periodicity
from nsfrb.flagging import flag_vis
#from nsfrb.planning import find_fast_vis_label
from nsfrb.config import tsamp_slow,fmin,fmax,nchans,NUM_CHANNELS, CH0, CH_WIDTH, AVERAGING_FACTOR, IMAGE_SIZE, c, Lon,Lat, DM_tol_slow,DM_tol,table_dir,tsamp_imgdiff,candplotfile_slow,candplotfile_imgdiff,candplotfile,img_dir,freq_axis,freq_axis_fullres,raw_cand_dir,bad_antennas,flagged_antennas,lambdaref,pixperFWHM,remote_cand_dir,minDM,maxDM,fc,chanbw,final_cand_dir
from nsfrb.config import tsamp as tsamp_ms
from nsfrb import plotting as pl
from nsfrb import pipeline
from event import names
import csv
from nsfrb.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset
from nsfrb.classifying_with_time import classify_images_3D
from concurrent.futures import ThreadPoolExecutor
from nsfrb.imaging import uv_to_pix,single_pix_image
from nsfrb.noise import noise_update_all
from nsfrb.planning import gen_dm
import argparse
from dsautils import dsa_store
from nsfrb import candcutting as cc
import json
import os
import sys
import numpy as np
from astropy.time import Time
from nsfrb.config import baseband_tsamp,tsamp_slow,tsamp_imgdiff
from nsfrb.config import nsamps as init_nsamps
from nsfrb.outputlogging import printlog,send_candidate_pushover,send_candidate_slack
from nsfrb import candcutting
from event import event
from dsaT4 import data_manager
from dask.distributed import Client#, Lock
from dask.distributed import Lock as Lock_DASK
from threading import Lock
from copy import deepcopy
from itertools import chain
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import MappingProxyType
from typing import Union

from astropy.time import Time
import astropy.units as u
from dsautils import cnf
from dsautils import dsa_syslog as dsl
"""
Functions for converting cand cutter output into T4 triggers and json files
compatible with DSA-110 event scheduler
"""

#client = Client('10.41.0.254:8781')
#client = ThreadPoolExecutor(40)#Client('tcp://10.42.0.228:8786')#10.42.0.232:8786')
LOCK = None #Lock('update_json')
widthtrials = np.array(2**np.arange(5),dtype=int)
DM_trials = np.array(gen_dm(minDM,maxDM,DM_tol,fc*1e-3,nchans,tsamp_ms,chanbw,init_nsamps))    
DM_trials_slow = np.array(gen_dm(minDM*5,maxDM*5,DM_tol,fc*1e-3,nchans,tsamp_slow,chanbw,init_nsamps))
tDM_max = (4.15)*np.max(DM_trials)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms
tDM_max_slow = (4.15)*np.max(DM_trials_slow)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms
maxshift = int(np.ceil(tDM_max/tsamp_ms))
maxshift_slow = int(np.ceil(tDM_max_slow/tsamp_slow))
ds = dsa_store.DsaStore()
"""
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsaT4")
LOGGER.function("T4_manager")
#dc = alert_client.AlertClient('dsa')

TIMEOUT_FIL = 600
"""
FILPATH = os.environ["DSA110DIR"] + "operations/T1/"
OUTPUT_PATH = os.environ["DSA110DIR"] + "operations/T4/"
#IP_GUANO = '3.13.26.235'

def nsfrb_to_json(cand_isot,mjds,snr,width,dm,ra,dec,trigname,P=-1,final_cand_dir=final_cand_dir,slow=False,imgdiff=False):
    """
    Takes the following arguments and saves to a json file in the specified cand dir
    cand_isot: str
    snr: float
    width: int
    dm: float
    ra: float
    dec: float
    trigname: str
    """
    #mjds = Time(cand_isot,format='isot').mjd
    if slow:
        ibox = int(np.ceil(width*tsamp_slow/baseband_tsamp))
    elif imgdiff:
        ibox = int(np.ceil(width*tsamp_imgdiff/baseband_tsamp))
    else:
        ibox = int(np.ceil(width*tsamp_ms/baseband_tsamp))
    f = open(final_cand_dir + "/" + trigname + ".json","w")
    json.dump({"mjds":mjds,
               "isot":cand_isot,
               "snr":snr,
               "ibox":ibox,
               "dm":dm,
               "ibeam":-1,
               "cntb":-1,
               "cntc":-1,
               "specnum":-1,
               "ra":ra,
               "dec":dec,
               "trigname":trigname,
               "period":P
               },f)
    f.close()

    return final_cand_dir + "/" + trigname + ".json"


#LOCK = Lock('update_json')
from nsfrb.config import cutterfile
from simulations_and_classifications import generate_PSF_images as scPSF
def cluster_manage(d_future,image,nsamps,dec_obs,args,cutterfile,DM_trials_use,widthtrials,injection_flag,postinjection_flag,PSF):
    if len(args.daskaddress) > 0:
        raw_cand_names,finalcands = d_future
    else:
        raw_cand_names,finalcands = d_future.result()
    if not (args.cluster and len(finalcands)>=args.mincluster):
        #cut by S/N if still too many
        if args.maxcand:
            printlog("Identifying max S/N candidate",output_file=cutterfile)
            sortedcands = list(np.array(finalcands)[np.argsort(np.array(finalcands)[:,-1])[::-1],:])
            finalcands = sortedcands[0:1]
            finalidxs = np.arange(1)
        elif len(finalcands) >args.maxcands_postcluster:
            printlog(cand_isot + "has too many candidates to process post-clustering (" + str(len(finalcands)) + ">" + str(args.maxcands_postcluster) + ") limit...",output_file=cutterfile)
            sortedcands = list(np.array(finalcands)[np.argsort(np.array(finalcands)[:,-1])[::-1],:])
            finalcands = sortedcands[:int(args.maxcands_postcluster)]
            finalidxs = np.arange(len(finalcands),dtype=int)
            printlog("done, cut to " + str(len(finalcands)) + " candidates",output_file=cutterfile)


        return finalcands,finalidxs
    
    printlog("PRE-CLUSTERING THERE ARE " + str(len(finalcands)) + " CANDIDATES",output_file=cutterfile)
    finalidxs = np.arange(len(finalcands),dtype=int)
    useTOA=args.useTOA and len(finalcands[0])==6
    #start clustering
    printlog("clustering with HDBSCAN...",output_file=cutterfile)
    #clustering with hdbscan
    """
    if args.psfcluster:
        PSF,PSF_params = scPSF.manage_PSF(scPSF.make_PSF_dict(),(2*image.shape[0])+1,dec_obs,nsamps=nsamps)#scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),image.shape[0],True,nsamps).mean((2,3))
        PSF = PSF.mean((2,3))
        printlog("PSF shape for clustering:" + str(PSF.shape),output_file=cutterfile)
    else:
        PSF = None
    """
    for i in range(args.clusteriters):
        mincluster = int(np.max([args.mincluster//(i+1),2]))

        printlog("Cluster iteration " + str(i+1) + "/" + str(args.clusteriters) + " with min cluster size " + str(mincluster),output_file=cutterfile)
        if useTOA:
            classes,cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs,centroid_TOAs = cc.hdbscan_cluster(finalcands,min_cluster_size=mincluster,min_samples=args.minsamples,dmt=DM_trials_use,wt=widthtrials,plot=False,show=False,SNRthresh=args.SNRthresh,PSF=(PSF if i==0 else None),useTOA=True,perc=args.psfpercentile,avgcluster=args.avgcluster)
        else:
            classes,cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs = cc.hdbscan_cluster(finalcands,min_cluster_size=mincluster,min_samples=args.minsamples,dmt=DM_trials_use,wt=widthtrials,plot=False,show=False,SNRthresh=args.SNRthresh,PSF=(PSF if i==0 else None),perc=args.psfpercentile,avgcluster=args.avgcluster)
        if np.all(np.array(classes)==-1):
            printlog("Minimum number of clusters reached",output_file=cutterfile)
            break
        else:
            printlog("done, made " + str(len(cluster_cands)) + " clusters",output_file=cutterfile)
            printlog(classes,output_file=cutterfile)
            printlog(cluster_cands,output_file=cutterfile)

            finalcands = cluster_cands

    finalidxs = np.arange(len(finalcands),dtype=int)

    printlog("AFTER CLUSTERING THERE ARE " + str(len(finalidxs)) + " CANDIDATES",output_file=cutterfile)

    #cut by S/N if still too many
    if args.maxcand:
        printlog("Identifying max S/N candidate",output_file=cutterfile)
        sortedcands = list(np.array(finalcands)[np.argsort(np.array(finalcands)[:,-1])[::-1],:])
        finalcands = sortedcands[0:1]
        finalidxs = np.arange(1)
    elif len(finalcands) >args.maxcands_postcluster:
        printlog(cand_isot + "has too many candidates to process post-clustering (" + str(len(finalcands)) + ">" + str(args.maxcands_postcluster) + ") limit...",output_file=cutterfile)
        sortedcands = list(np.array(finalcands)[np.argsort(np.array(finalcands)[:,-1])[::-1],:])
        finalcands = sortedcands[:int(args.maxcands_postcluster)]
        finalidxs = np.arange(len(finalcands),dtype=int)
        printlog("done, cut to " + str(len(finalcands)) + " candidates",output_file=cutterfile)
    
    return finalcands,finalidxs

#ffa_semaphore = False
def ffa_manage(d_future,image,nsamps,nchans,dec_obs,args,cutterfile,DM_trials_use,widthtrials,cand_isot,injection_flag,postinjection_flag,slow,imgdiff,RA_axis_2D,DEC_axis_2D,tsamp_use,ffalock,suff):
    from nsfrb.planning import find_fast_vis_label
    from nsfrb.config import vis_dir
    #global ffa_semaphore
    tsamp_use = tsamp_ms
    if len(args.daskaddress) > 0:
        ret = d_future
    else:
        ret = d_future.result()
    #while ffa_semaphore:
    #    time.sleep(1)
    #ffa_semaphore = True
    #printlog("SEMAPHORE ACQUIRED",output_file=cutterfile)

    if ret is None:
        return None
    if len(ret)==4:
        finalcands,finalidxs,predictions,probabilities = ret
        classify_flag = True
    elif len(ret)==2:
        finalcands,finalidxs = ret
        classify_flag = False
    else:
        #ffa_semaphore = False
        return None

    if (not args.FFA) or args.completeness or args.remote:# or slow or imgdiff:
        #ffa_semaphore = False
        if classify_flag:
            return finalcands,finalidxs,dict(),predictions,probabilities 
        else:
            return finalcands,finalidxs,dict()

    printlog("FFA START",output_file=cutterfile)

    #get path
    if args.GP:
        gppaths = glob.glob(vis_dir + "/GP_observations_*")
        gpisots_ = np.sort([gppaths[i][-23:] for i in range(len(gppaths))])
        gptimes = []
        for i in range(len(gpisots_)):
            try:
                gptimes.append(Time(gpisots_[i],format='isot'))
            except:
                printlog("Skipping " + gpisots_[i],output_file=cutterfile)
        obstime = Time(cand_isot,format='isot')
        printlog(gptimes,output_file=cutterfile)
        for i in range(len(gptimes)):
            if obstime<gptimes[i]:
                break
        printlog("HERE " + gptimes[i-1].isot + " " + obstime.isot,output_file=cutterfile) 
        fpath = vis_dir + "/GP_observations_" + gptimes[i-1].isot + "/"
    else:
        fpath = ""
    printlog(fpath,output_file=cutterfile)


    #find the data file
    printlog("HERE",output_file=cutterfile)
    printlog("candisot:" + cand_isot,output_file=cutterfile)
    printlog("path:" + fpath,output_file=cutterfile)
    fnum,offset,dec = find_fast_vis_label(Time(cand_isot,format='isot').mjd,return_dec=True,path=fpath)
    printlog("FFVL OUTPUT:" + str((fnum,offset,dec)),output_file=cutterfile)
    if offset == -1:
        #ffa_semaphore = False
        if classify_flag:
            return finalcands,finalidxs,dict(),predictions,probabilities
        else:
            return finalcands,finalidxs,dict()
    candgulp = int(offset//25)
    ngulps = args.FFAgulps
    mingulp = max([0,candgulp-(ngulps//2)])
    maxgulp = min([90,mingulp + ngulps])
    ngulps = maxgulp-mingulp
    printlog("periodicity searching gulps " + str(mingulp) + "-" + str(maxgulp) + " from file " + str(fnum),output_file=cutterfile)
    gulpsize = 25
    if args.GP:
        fname = fpath + "/nsfrb_sb00_" + str(fnum) +".out"
    else:
        fname = vis_dir + "/lxd110h03/nsfrb_sb00_" + str(fnum) +".out"
    printlog(fname,output_file=cutterfile)
    nchan_per_node=nchans_per_node = 8
    sb,mjd,dec = pipeline.read_raw_vis(fname,nchan=nchan_per_node,nsamps=gulpsize,gulp=0,headersize=16,get_header=True)
    printlog("file params:" + str((sb,mjd,dec)),output_file=cutterfile)

    #target
    printlog(finalcands,output_file=cutterfile)
    printlog(finalidxs,output_file=cutterfile)
    target_ras = RA_axis_2D[np.array([int(finalcands[j][1]) for j in finalidxs],dtype=int),np.array([int(finalcands[j][0]) for j in finalidxs],dtype=int)]
    target_decs = DEC_axis_2D[np.array([int(finalcands[j][1]) for j in finalidxs],dtype=int),np.array([int(finalcands[j][0]) for j in finalidxs],dtype=int)]
    printlog(target_ras,output_file=cutterfile)
    printlog(target_decs,output_file=cutterfile)
    target_coords = SkyCoord(ra=target_ras*u.deg,
                            dec=target_decs*u.deg,frame='icrs')
    target_dms = DM_trials_use[np.array([int(finalcands[j][3]) for j in finalidxs],dtype=int)]


    sbs=["0"+str(p) if p < 10 else str(p) for p in range(16)]
    corrs = ["h03","h04","h05","h06","h07","h08","h10","h11","h12","h14","h15","h16","h18","h19","h21","h22"]
    nbin = args.FFAbin
    if slow:
        nbin *= int(tsamp_slow//tsamp_ms)
    elif imgdiff:
        nbin *= int(tsamp_imgdiff//tsamp_ms)
    image_size = image.shape[0]
    printlog("target coords:" + str(target_coords),output_file=cutterfile)


    #image
    ffalock.acquire()
    printlog("MUTEX ACQUIRED",output_file=cutterfile)    
    alldspec = np.zeros((len(finalcands),gulpsize*ngulps//nbin,(1 if args.FFAbinchans else nchan_per_node)*nchans))
    allpix_all = [[]]*len(finalcands)
    printlog("start imaging..." , output_file=cutterfile)
    for j in range(nchans):
        printlog("sb " + str(j),output_file=cutterfile)
        if args.GP:
            fname = fpath + "/nsfrb_sb"+sbs[j]+"_" + str(fnum) +".out"
        else:
            fname = vis_dir + "/lxd110"+corrs[j]+"/nsfrb_sb"+sbs[j]+"_" + str(fnum) +".out"
        try:
            dat_all,sb,mjd,dec = pipeline.read_raw_vis(fname,gulp=mingulp,nsamps=gulpsize*ngulps,nchan=nchan_per_node,headersize=16,get_header=False)
        except Exception as exc:
            printlog(fname + " not found: " + str(exc),output_file=cutterfile)
            if args.FFAbinchans:
                alldspec[:,:,j] = np.nan
            else:
                alldspec[:,:,j*nchans_per_node:(j+1)*nchans_per_node] = np.nan
            continue
        test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None)
        pt_dec = dec*np.pi/180.
        bname, blen, UVW = pu.baseline_uvw(antenna_order, pt_dec, refmjd, casa_order=False)
        outriggers=False
        dat_all, bname, blen, UVW, antenna_order = flag_vis(dat_all, bname, blen, UVW, antenna_order,
                                                    bad_antennas,
                                                    bmin=20,flagged_corrs=[],flag_channel_templates=[],
                                                    flagged_chans=[],flagged_baseline_idxs=[])
        U = UVW[0,:,1]
        V = UVW[0,:,0]
        W = UVW[0,:,2]
        nchans = 16
        fobs = (np.reshape(freq_axis_fullres,(nchans*nchans_per_node,int(NUM_CHANNELS/2/nchans_per_node))).mean(axis=1))*1e-3

        if j == 0:
            uv_diag=np.max(np.sqrt(U**2 + V**2))
            pixel_resolution = (lambdaref / uv_diag) / pixperFWHM
        
        for i in range(len(target_coords)):
            target_coord = target_coords[i]
            ret = single_pix_image(dat_all,U,V,fobs,j,dec,mjd,ngulps,nbin,target_coord,tsamp_use,pixel_resolution,pixperFWHM,uv_diag,nchans_per_node=8,allpix=allpix_all[i],DM=target_dms[i])
            if args.FFAbinchans:
                alldspec[i,:,j] = np.nanmean(ret[0],1)
            else:
                alldspec[i,:,j*nchans_per_node:(j+1)*nchans_per_node] = ret[0]
            allpix_all[i] = ret[1]
        del dat_all
        printlog("sb done",output_file=cutterfile)

    for i in range(len(target_coords)):
        np.save(raw_cand_dir + "/single_pix_" + cand_isot + "_" + str(i) + suff + ".npy",alldspec[i,:,:])
    printlog("done creating single pix dynamic spectra",output_file=cutterfile)
    ffalock.release()
    printlog("starting periodicity search...",output_file=cutterfile)
    trial_periods = np.array(args.periods)
    trial_periods = trial_periods[trial_periods<alldspec.shape[1]]
    finalPcands = []
    finalPidxs = []
    #get the current noise map from file
    prev_noise_,prev_noise_N = noise_update_all(None,image_size,image_size,DM_trials_use,widthtrials,readonly=True) #noise.get_noise_dict(gridsize,gridsize)
    prev_noise = prev_noise_[0,0]/np.sqrt(nbin)
    finalpcands = dict()
    for i in range(len(target_coords)):
        dspec = alldspec[i,:,:]
        timeseries = np.nanmean((dspec - np.nanmedian(dspec,axis=0)),1)
        timeseries /= prev_noise #np.nanstd(timeseries)
        snrs = periodicity.ffa_slow(timeseries,trial_periods)
        printlog(snrs,output_file=cutterfile)
        if np.max(snrs)>args.FFASNRthresh:
            peakidx = np.unravel_index(np.argmax(snrs),snrs.shape)
            printlog("found candidate pulse period P=" + str(trial_periods[peakidx[0]]*tsamp_use*nbin/1000) + " s",output_file=cutterfile) 
            trial_p_fine = np.linspace(trial_periods[max([peakidx[0]-1,0])],trial_periods[min([peakidx[0]+1,len(trial_periods)-1])],10)
            resids = periodicity.ffa_timing(timeseries,trial_p_fine,trial_periods[peakidx[0]])
            minresid = np.unravel_index(np.argmin(resids),resids.shape)
            printlog("refined pulse period P=" + str(trial_p_fine[minresid[0]]*tsamp_use*nbin/1000) + " s",output_file=cutterfile)
            #finalPcands.append(list(finalcands[i]) + [trial_p_fine[minresid[0]]*tsamp_use*nbin/1000])
            #finalPidxs.append(finalidxs[i])
            finalpcands[i] = dict()
            finalpcands[i]["snrs"] = snrs
            finalpcands[i]["resids"] = resids
            finalpcands[i]["initP_samps"] = trial_periods[peakidx[0]]
            finalpcands[i]["initP_secs"] = trial_periods[peakidx[0]]*tsamp_use*nbin/1000
            finalpcands[i]["fineP_samps"] = trial_p_fine[minresid[0]]
            finalpcands[i]["fineP_secs"] = trial_p_fine[minresid[0]]*tsamp_use*nbin/1000
            finalpcands[i]["timeseries"] = timeseries
            finalpcands[i]["taxis"] = np.arange(len(timeseries))
            finalpcands[i]["trial_p_cand_secs"] = trial_p_fine*tsamp_use*nbin/1000
            printlog("written to dict",output_file=cutterfile)
        else:
            printlog("none found",output_file=cutterfile)
    #ffa_semaphore = False
    #ffalock.release()
    if classify_flag:
        return finalcands,finalidxs,finalpcands,predictions,probabilities
    else:
        return finalcands,finalidxs,finalpcands

def classify_manage(d_future,image,nsamps,nchans,dec_obs,args,cutterfile,DM_trials_use,widthtrials,cand_isot,injection_flag,postinjection_flag,slow,imgdiff):
    if len(args.daskaddress) > 0:
        finalcands,finalidxs = d_future
    else:
        finalcands,finalidxs = d_future.result()
    printlog("CLASSIFY STARTED",output_file=cutterfile)
    classify_flag = (args.classify or args.classify3D)
    useTOA=args.useTOA and len(finalcands[0])==6
    if not classify_flag:
        return finalcands,finalidxs
    
    if args.classify:

        if args.subimgpix == image.shape[0]:
            printlog(str("IMGDIFF: " if imgdiff else "") + "Using full image for classification and cutouts;"+str(image.shape),output_file=cutterfile)
            data_array = (cc.img_to_classifier_format(np.repeat(image.mean(2),nchans,axis=2),cand_isot,img_dir)[np.newaxis,:,:,:]).repeat(len(finalcands),axis=0)
        else:
            #make a binned copy for each candidate
            data_array = np.zeros((len(finalcands),args.subimgpix,args.subimgpix,image.shape[3]),dtype=np.float64)
            for j in range(len(finalcands)):
                printlog(finalcands[j],output_file=cutterfile)

                #don't need to dedisperse(?)
                subimg = cc.get_subimage(image,int(finalcands[j][0]),int(finalcands[j][1]),save=False,subimgpix=args.subimgpix)
                if useTOA:
                    printlog("using TOA...",output_file=cutterfile)
                    loc = int(finalcands[j][4])
                    printlog("got loc...",output_file=cutterfile)
                    wid = widthtrials[int(finalcands[j][2])]
                    printlog("got wid...",output_file=cutterfile)
                    data_array[j,:,:,:] = cc.img_to_classifier_format(subimg[:,:,int(loc+1-(wid//2)):int(loc+1-(wid//2) + wid),:].mean(2),cand_isot+"_"+str(j),img_dir)
                    printlog("img to classifier formatd done...",output_file=cutterfile)
                else:
                    data_array[j,:,:,:] = cc.img_to_classifier_format(subimg.mean(2),cand_isot+"_"+str(j),img_dir)  #.mean(2)#subimg[:,:,np.argmax(subimg.sum((0,1,3))),:]
                printlog("cand shape:" + str(data_array[j,:,:,:].shape),output_file=cutterfile)

        #reformat for classifier
        #transposed_array = np.transpose(data_array, (0,3,1,2))#cands x frequencies x RA x DEC
        #new_shape = (data_array.shape[0], data_array.shape[3], data_array.shape[1], data_array.shape[2])
        merged_array = np.transpose(data_array, (0,3,1,2)) #transposed_array.reshape(new_shape)

        printlog("shape input to classifier:" + str(merged_array.shape),output_file=cutterfile)
        #run classifier
        predictions, probabilities = classify_images(merged_array, args.model_weights, verbose=args.verbose)
        printlog(predictions,output_file=cutterfile)
        printlog(probabilities,output_file=cutterfile)

        #only save bursts likely to be real
        #finalidxs = finalidxs[~np.array(predictions,dtype=bool)]

    elif args.classify3D:
        if args.subimgpix == image.shape[0]:
            printlog("Using full image for classification and cutouts",output_file=cutterfile)
            if imgdiff:
                if useTOA:
                    tmpsnrs = np.array([fcand[-1] for fcand in finalcands])
                    tmptoa = np.array([fcand[4] for fcand in finalcands])[np.argmax(tmpsnrs)]
                    printlog("so far so good",output_file=cutterfile)
                    if tmptoa < init_nsamps//2:
                        data_array = (image[np.newaxis,:,:,:init_nsamps,:].repeat(nchans,axis=4)).repeat(len(finalcands),axis=0)
                    else:
                        data_array = (image[np.newaxis,:,:,-init_nsamps:,:].repeat(nchans,axis=4)).repeat(len(finalcands),axis=0)
                    printlog("still ok",output_file=cutterfile)
                else:
                    data_array = (image[np.newaxis,:,:,:init_nsamps,:].repeat(nchans,axis=4)).repeat(len(finalcands),axis=0)
            else:
                data_array = (image[np.newaxis,:,:,:,:]).repeat(len(finalcands),axis=0)
        else:
            #make a binned copy for each candidate
            data_array = np.zeros((len(finalcands),args.subimgpix,args.subimgpix,image.shape[2],image.shape[3]),dtype=np.float32)
            for j in range(len(finalcands)):
                printlog(finalcands[j],output_file=cutterfile)

                #don't need to dedisperse(?)
                data_array[j,:,:,:,:] = cc.get_subimage(image,int(finalcands[j][0]),int(finalcands[j][1]),save=False,subimgpix=args.subimgpix)
                printlog("cand shape:" + str(data_array[j,:,:,:].shape),output_file=cutterfile)

        #run classifier
        printlog("still fine",output_file=cutterfile)
        printlog("Start classifying " + str(data_array.shape),output_file=cutterfile)
        predictions, probabilities = classify_images_3D(data_array, args.model_weights3D, verbose=args.verbose)
        if args.testtrigger:
            predictions = np.zeros(len(finalcands))
            probabilities = np.zeros(len(finalcands))
        printlog(predictions,output_file=cutterfile)
        printlog(probabilities,output_file=cutterfile)

        #only save bursts likely to be real
        #finalidxs = finalidxs[~np.array(predictions,dtype=bool)]


    #if set, cut out candidates rejected by the classifier
    if classify_flag and args.classcut:
        printlog("Classifier rejected " + str(np.sum(predictions)) + "/" + str(len(predictions)) + " candidates",output_file=cutterfile)
        finalcands_new = []
        for i in range(len(finalcands)):
            if predictions[i] == 0:
                finalcands_new.append(finalcands[i])
        finalcands = finalcands_new
        if len(finalcands) == 0:
            printlog("No remaining candidates, done",output_file=cutterfile)
            if args.remote:
                os.system("ssh h24.pro.pvt \"rm " + raw_cand_dir + "*" + cand_isot + "*\"")
            else:
                os.system("rm " + raw_cand_dir + "*" + cand_isot + "*")
            return
        probabilities = probabilities[predictions==0]
        predictions = predictions[predictions==0]
        finalidxs = np.arange(len(finalcands),dtype=int)
    return finalcands,finalidxs,predictions,probabilities

def writecands_manage(d_future,image,args,DM_trials_use,widthtrials,suff,cand_isot,cand_mjd,slow,imgdiff,injection_flag,postinjection_flag,tsamp_use,nsamps,RA_axis_2D,DEC_axis_2D,cutterfile):
    if len(args.daskaddress) > 0:
        res = d_future
    else:
        res = d_future.result()
    print("RIGHT HERE",res)
    if res is None:
        return
    elif len(res) == 5:
        finalcands,finalidxs,finalpcands,predictions,probabilities = res
        classify_flag = True
    elif len(res) == 3:
        finalcands,finalidxs,finalpcands = res
        classify_flag = False
    else:
        return
    print("done parsing result")
    
    #if its an injection write the highest SNR candidate to the injection tracker
    useTOA=args.useTOA and len(finalcands[0])==6
    #injection_flag,postinjection_flag = cc.is_injection(cand_isot)
    if injection_flag:
        with open(recover_file,"a") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            for j in finalidxs:
                wr.writerow([cand_isot,DM_trials_use[int(finalcands[j][3])],widthtrials[int(finalcands[j][2])],finalcands[j][-1],(None if not classify_flag else predictions[j]),(None if not classify_flag else probabilities[j])])
        csvfile.close()

        if args.remote:
            printlog("updating injection files on h24...",output_file=cutterfile)
            os.system("scp "+recover_file+" h24.pro.pvt:/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-injections/")
            printlog("done",output_file=cutterfile)

    print("done updating recoveries")
    #make final directory for candidates
    dirlabel = "candidates"
    if injection_flag:
        dirlabel = "injections"
    elif args.completeness:
        dirlabel = "completeness"

    if args.remote:
        os.system("ssh h24.pro.pvt \"mkdir "+ final_cand_dir + dirlabel + "/" + cand_isot + suff+"\"")
    else:
        os.system("mkdir "+ final_cand_dir + dirlabel + "/" + cand_isot + suff)

    #write final candidates to csv
    prefix = "NSFRB"
    with open(table_dir+"nsfrb_lastname.txt","r") as lnamefile:
        lastname = (lnamefile.read()).strip()
        if lastname == "None":
            lastname = None
    lnamefile.close()
    print("done getting lastname")

    #lastname =      #once we have etcd, change to 'names.get_lastname()'
    allcandnames = []
    if args.remote:
        csvfile = open(remote_cand_dir + "/final_candidates_" + cand_isot + ".csv","w")
    else:
        csvfile = open(final_cand_dir+ dirlabel  + "/" + cand_isot + suff + "/final_candidates_" + cand_isot + ".csv","w")
    wr = csv.writer(csvfile,delimiter=',')
    hdr = ["candname","RA index","DEC index","WIDTH index", "DM index"]
    if useTOA: hdr += ["TOA"]
    if (args.FFA and not args.completeness): hdr += ["Period_s"]
    hdr += ["SNR"]
    if classify_flag: hdr += ["PROB"]
    wr.writerow(hdr)
    sysstdout = sys.stdout
    for j in finalidxs:#range(len(finalidxs)):
        with open(cutterfile,"a") as sys.stdout:
            lastname = names.increment_name(cand_mjd,lastname=lastname)
        sys.stdout = sysstdout
        if classify_flag:
            if (args.FFA and not args.completeness): #args.FFA:
                wr.writerow(np.concatenate([[lastname],np.array(finalcands[j][:-1],dtype=int),["" if j not in finalpcands.keys() else finalpcands[j]["fineP_secs"]],[finalcands[j][-1]],[probabilities[j]]]))
            else:
                wr.writerow(np.concatenate([[lastname],np.array(finalcands[j][:-1],dtype=int),[finalcands[j][-1]],[probabilities[j]]]))
        else:
            if (args.FFA and not args.completeness): #args.FFA:
                wr.writerow(np.concatenate([[lastname],np.array(finalcands[j][:-1],dtype=int),["" if j not in finalpcands.keys() else finalpcands[j]["fineP_secs"]],[finalcands[j][-1]]]))
            else:
                wr.writerow(np.concatenate([[lastname],np.array(finalcands[j][:-1],dtype=int),[finalcands[j][-1]]]))
        allcandnames.append(prefix + lastname)
    csvfile.close()
    if args.remote:
        printlog("copying cand file to h24...",output_file=cutterfile)
        os.system("scp "+remote_cand_dir + "/final_candidates_" + cand_isot + ".csv h24.pro.pvt:"+final_cand_dir+ dirlabel  + "/" + cand_isot + suff + "/")
        printlog("done")

    with open(table_dir+"nsfrb_lastname.txt","w") as lnamefile:
        if lastname is not None:
            lnamefile.write(lastname)
        else:
            lnamefile.write("None")
    lnamefile.close()
    printlog("done naming stuff",output_file=cutterfile)

    print("dones writing to csv")
    print(finalidxs,finalcands)
    #make subdirectories for candidates
    for j in finalidxs:

        lastname = allcandnames[j]
        #make folder for each candidate
        if args.remote:
            os.system("ssh h24.pro.pvt \"mkdir "+ final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + lastname+"\"")
            os.system("ssh h24.pro.pvt \"mkdir "+ final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + lastname + "/voltages\"")
        else:
            os.system("mkdir "+ final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + lastname)
            os.system("mkdir "+ final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + lastname + "/voltages")

    if len(finalidxs) > 0:
        #make diagnostic plot
        printlog("making diagnostic plot...",output_file=cutterfile,end='')
        canddict=dict()
        canddict['ra_idxs'] = [finalcands[j][0] for j in finalidxs]
        canddict['dec_idxs'] = [finalcands[j][1] for j in finalidxs]
        canddict['wid_idxs'] = [finalcands[j][2] for j in finalidxs]
        canddict['dm_idxs'] = [finalcands[j][3] for j in finalidxs]
        canddict['snrs'] = [finalcands[j][-1] for j in finalidxs]
        printlog("SNRS:" + str(canddict['snrs']),output_file=cutterfile)
        canddict['names'] = allcandnames
        if classify_flag:
            canddict['probs'] = probabilities
            canddict['predicts'] = predictions
        if useTOA:
            canddict['TOAs'] = [finalcands[j][4] for j in finalidxs]
            printlog("TOAS:" + str(canddict['TOAs']),output_file=cutterfile)
        #RA_axis,DEC_axis,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs)

        # dedisperse to each unique dm candidate
        timeseries = []
        #sourceimg_all = np.concatenate([np.zeros(tuple(list(image.shape[:2])+[0 if (slow or imgdiff) else maxshift]+[image.shape[3]])),image],axis=2)
        for i in range(len(finalidxs)):
            print(i)
            DM = DM_trials_use[int(canddict['dm_idxs'][i])]

            sourceimg = image[int(canddict['dec_idxs'][i]):int(canddict['dec_idxs'][i])+1,
                                    int(canddict['ra_idxs'][i]):int(canddict['ra_idxs'][i])+1,:,:]#np.concatenate([np.zeros((1,1,maxshift,image.shape[3])),image[canddict['dec_idxs'][i],canddict['ra_idxs'][i],:,:],axis=2)
            if (DM != 0 and not imgdiff):
                printlog("COMPUTING SHIFTS FOR DM="+str(DM)+"pc/cc "+ str(sourceimg.shape),output_file=cutterfile)

                tshift =np.array(np.abs((4.15)*DM*((1/np.nanmin(freq_axis)/1e-3)**2 - (1/freq_axis/1e-3)**2))//tsamp_use,dtype=int)
                sourceimg_dm = np.zeros_like(sourceimg)
                for j in range(len(freq_axis)):
                    sourceimg_dm[:,:,:,j] = np.pad(sourceimg[:,:,:,j],((0,0),(0,0),(tshift[j],0)),mode='constant')[:,:,:sourceimg.shape[2]]
                """


                if slow: 
                    maxshift_use = maxshift_slow
                else: 
                    maxshift_use = maxshift 
                corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append,wraps_append,wraps_no_append = gen_dm_shifts(np.array([DM]),freq_axis,tsamp_use,nsamps,outputwraps=True,maxshift=maxshift_use)#0 if (slow or imgdiff) else maxshift)

                printlog("corr shifts shape:" + str(corr_shifts_all_no_append.shape),output_file=cutterfile)

                DM_idx = 0#list(DM_trials).index(DM)
                printlog("PRE-DM SHAPE:"+str(sourceimg.shape),output_file=cutterfile)
                sourceimg_dm = (((((np.take_along_axis(sourceimg[:,:,:,np.newaxis,:].repeat(1,axis=3).repeat(2,axis=4),indices=corr_shifts_all_no_append[:,:,:,DM_idx:DM_idx+1,:],axis=2))*tdelays_frac_no_append[:,:,:,DM_idx:DM_idx+1,:]))[:,:,:,0,:]))
                printlog("POST-DM SHAPE:"+str(sourceimg_dm.shape),output_file=cutterfile)
                #zero out anywhere that was wrapped
                #sourceimg_dm[wraps_no_append[:,:,:,DM_idx,:].repeat(sourceimg.shape[0],axis=0).repeat(sourceimg.shape[1],axis=1)] = 0

                #now average the low and high shifts 
                sourceimg_dm = (sourceimg_dm.reshape(tuple(list(sourceimg.shape)[:2] + [nsamps,nchans] + [2])).sum(4))
                """
            else:
                sourceimg_dm = sourceimg
            timeseries.append(np.nanmean(sourceimg_dm,(0,1,3)))

            print(injection_flag,canddict)
            if not injection_flag:
                print("IN THIS LOOP FOR SOME REASON")
                #create json file
                snr=canddict['snrs'][i]
                width=int(widthtrials[int(canddict['wid_idxs'][i])])
                print("HERE")
                dm=DM_trials_use[int(canddict['dm_idxs'][i])]
                print(dm,canddict['dec_idxs'],canddict['ra_idxs'],RA_axis_2D.shape)
                ra=RA_axis_2D[int(canddict['dec_idxs'][i]),int(canddict['ra_idxs'][i])] #RA_axis[int(canddict['ra_idxs'][i])]
                print("HEREHERE")
                dec=DEC_axis_2D[int(canddict['dec_idxs'][i]),int(canddict['ra_idxs'][i])] #DEC_axis[int(canddict['dec_idxs'][i])]
                print("HALO")
                trigname = canddict['names'][i]
                printlog(str(snr) +","+ str(width)+","+str(dm) + ","+ str(ra) + "," + str(dec) + "," + trigname,output_file=cutterfile)
                print("ALMOST DONE")
                if useTOA:
                    toa = canddict['TOAs'][i]
                    cand_mjd = Time(Time(cand_isot,format='isot').mjd + (canddict['TOAs'][i]*(tsamp_use)/1000/86400),format='mjd').mjd
                else:
                    cand_mjd = Time(cand_isot,format='isot').mjd
                print("ALMOST ALMOST DONE",finalpcands)
                if (args.FFA and not args.completeness and not args.remote) and (i in finalpcands.keys()):
                    P = float(finalpcands[i]["fineP_secs"])
                else:
                    P =-1
                print("RIGHT HERE:",finalpcands,i,P)
                if args.remote:
                    fl = nsfrb_to_json(cand_isot,cand_mjd,snr,width,dm,ra,dec,trigname,P=P,final_cand_dir=remote_cand_dir,slow=slow,imgdiff=imgdiff)
                    os.system("scp "+remote_cand_dir+trigname+".json h24.pro.pvt:"+final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + trigname + "/")
                else:
                    fl = nsfrb_to_json(cand_isot,cand_mjd,snr,width,dm,ra,dec,trigname,P=P,final_cand_dir=final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + trigname + "/",slow=slow,imgdiff=imgdiff)

                printlog(fl,output_file=cutterfile)
    
        print("writecands done")
        if not injection_flag:
            return list(res) + [canddict,allcandnames,timeseries,fl]
        else:
            return list(res) + [canddict,allcandnames,timeseries]
    return

def sendtrigger_manage(d_future,image,searched_image,args,uv_diag,dec_obs,slow,imgdiff,RA_axis,DEC_axis,DM_trials_use,widthtrials,cand_isot,suff,cutterfile,injection_flag,postinjection_flag,plotlock):
    if len(args.daskaddress)>0:
        res = d_future
    else:
        res = d_future.result()

    if res is None:
        return
    """
    if (not args.FFA) or args.completeness:
        if len(res) == 8:
            finalcands,finalidxs,predictions,probabilities,canddict,allcandnames,timeseries,jsonfname = res
            classify_flag = True
            #injection_flag = True
        elif len(res) == 7:
            finalcands,finalidxs,predictions,probabilities,canddict,allcandnames,timeseries = res
            classify_flag = True
            #injection_flag = False
        elif len(res) == 6:
            finalcands,finalidxs,canddict,allcandnames,timeseries,jsonfname = res
            classify_flag = False
            #injection_flag = True
        elif len(res) == 5:
            finalcands,finalidxs,canddict,allcandnames,timeseries = res
            classify_flag = False
            #injection_flag = False
        else:
            return
        finalpcands = dict()
    else:
    """
    if len(res) == 9:
        finalcands,finalidxs,finalpcands,predictions,probabilities,canddict,allcandnames,timeseries,jsonfname = res
        classify_flag = True
        #injection_flag = True
    elif len(res) == 8:
        finalcands,finalidxs,finalpcands,predictions,probabilities,canddict,allcandnames,timeseries = res
        classify_flag = True
        #injection_flag = False
    elif len(res) == 7:
        finalcands,finalidxs,finalpcands,canddict,allcandnames,timeseries,jsonfname = res
        classify_flag = False
        #injection_flag = True
    elif len(res) == 6:
        finalcands,finalidxs,finalpcands,canddict,allcandnames,timeseries = res
        classify_flag = False
        #injection_flag = False
    else:
        return
    
    dirlabel = "candidates"
    if injection_flag:
        dirlabel = "injections"
    elif args.completeness:
        dirlabel = "completeness"


    printlog("Creating candplot...",output_file=cutterfile)
    print(final_cand_dir + dirlabel + "/" + cand_isot + suff + "/")
    print(canddict,image,RA_axis,DEC_axis)
    plotlock.acquire()
    candplot=pl.search_plots_new(canddict,image,cand_isot,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                            DM_trials=DM_trials_use,widthtrials=widthtrials,
                                            output_dir=remote_cand_dir if args.remote else final_cand_dir + dirlabel + "/" + cand_isot + suff + "/",
                                            show=False,s100=args.SNRthresh/2,
                                            injection=injection_flag,vmax=np.nanmax(searched_image),vmin=args.SNRthresh,
                                            searched_image=searched_image,timeseries=timeseries,uv_diag=uv_diag,
                                            dec_obs=dec_obs,slow=slow,imgdiff=imgdiff,pcanddict=finalpcands,output_file=cutterfile)
    printlog(candplot,output_file=cutterfile)

    if args.toslack:
        printlog("sending plot to slack...",output_file=cutterfile)
        send_candidate_slack(candplot,filedir=remote_cand_dir if args.remote else final_cand_dir + dirlabel + "/" + cand_isot + suff + "/")
        #printlog("sending plot to pushover...",output_file=cutterfile)
        #send_candidate_pushover(candplot,filedir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/")
        printlog("done!",output_file=cutterfile)
        printlog("sending plot to custom webserver 9089...",output_file=cutterfile)
        
        if not args.remote:

            if slow:
                os.system("cp " + final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + candplot + " " + candplotfile_slow)
            elif imgdiff:
                os.system("cp " + final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + candplot + " " + candplotfile_imgdiff)
            else:
                os.system("cp " + final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + candplot + " " + candplotfile)
            printlog("sending notification via x11...",output_file=cutterfile)
            os.system("cp " + final_cand_dir + dirlabel + "/" + cand_isot + suff + "/" + candplot + " " + os.environ["NSFRBDIR"] + "/scripts/x11display.png")
            os.system("echo " + str((2/3) if imgdiff else 1) + " > "+ os.environ["NSFRBDIR"] + "/scripts/x11size.txt")
            os.system("echo " + candplot + " > "+ os.environ["NSFRBDIR"] + "/scripts/x11alertmessage.txt")
            printlog("done!",output_file=cutterfile)
    plotlock.release()
    if args.trigger:
        T4trigger = event.create_event(fl)
        return list(res) + [candplot, T4trigger]
    else:
        return list(res) + [candplot]

def archive_manage(d_future,cand_isot,suff,cutterfile,injection_flag,postinjection_flag):
    if args.completeness or (not args.archive) or 'NSFRBT4' not in os.environ.keys():
        if args.remote:
            printlog("Clearing tmp cand dir...",output_file=cutterfile)
            os.system("rm "+remote_cand_dir + "/" + cand_isot + "*"+suff+"*")
            printlog("done",output_file=cutterfile)
        return None
    if len(args.daskaddress)>0:
        res = d_future
    else:
        res = d_future.result()
    if res is None:
        if args.remote:
            printlog("Clearing tmp cand dir...",output_file=cutterfile)
            os.system("rm "+remote_cand_dir + "/" + cand_isot + "*"+suff+"*")
            printlog("done",output_file=cutterfile)
        return
    else:
        finalcands,finalidxs = res[0],res[1]

    #send final candidates to T4 because they will be removed from h24 when it runs out of space
    #make a new directory for timestamp on T4
    T4dir = os.environ['NSFRBT4']
    if injection_flag:
        T4dir += "injections"
    else:
        T4dir += "candidates"
    printlog("ssh user@dsa-storage.ovro.pvt \"mkdir "+ T4dir + "/" + cand_isot+ suff+"\"",output_file=cutterfile)
    os.system("ssh user@dsa-storage.ovro.pvt \"mkdir "+ T4dir + "/" + cand_isot+ suff+"\"")


    #copy csv and cand plot
    if args.remote:
        printlog("scp " + remote_cand_dir + "/" + cand_isot + "_NSFRBcandplot.png user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/",output_file=cutterfile)
        os.system("scp " + remote_cand_dir + "/" + cand_isot + "_NSFRBcandplot.png user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/")
        printlog("scp " + remote_cand_dir + "/"+ "final_candidates_" + cand_isot + ".csv user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/",output_file=cutterfile)
        os.system("scp " + remote_cand_dir + "/"+  "final_candidates_" + cand_isot + ".csv user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/")
    else:
        printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + cand_isot + "_NSFRBcandplot.png user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/",output_file=cutterfile)
        os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + cand_isot + "_NSFRBcandplot.png user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/")
        printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/"+ "final_candidates_" + cand_isot + ".csv user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/",output_file=cutterfile)
        os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/"+  "final_candidates_" + cand_isot + ".csv user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/")

    #make folder for each candidate
    printlog("ssh user@dsa-storage.ovro.pvt \"mkdir "+ " ".join([T4dir + "/" + cand_isot + suff + "/" + lastname for lastname in allcandnames]) + "\"",output_file=cutterfile)
    os.system("ssh user@dsa-storage.ovro.pvt \"mkdir "+ " ".join([T4dir + "/" + cand_isot + suff + "/" + lastname for lastname in allcandnames]) + "\"")
    printlog("ssh user@dsa-storage.ovro.pvt \"mkdir "+ " ".join([T4dir + "/" + cand_isot + suff + "/" + lastname + "/voltages/" for lastname in allcandnames]) + "\"",output_file=cutterfile)
    os.system("ssh user@dsa-storage.ovro.pvt \"mkdir "+ " ".join([T4dir + "/" + cand_isot + suff + "/" + lastname + "/voltages/" for lastname in allcandnames]) + "\"")
    for lastname in allcandnames:
        #copy numpy files
        if args.remote:
            printlog("scp " + remote_cand_dir + "/*" + " user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/" + lastname + "/",output_file=cutterfile)
            os.system("scp " + remote_cand_dir + "/*" + " user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/" + lastname + "/")
        else:
            printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname + "/*" + " user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/" + lastname + "/",output_file=cutterfile)
            os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname + "/*" + " user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/" + lastname + "/")

    if args.remote:
        printlog("Clearing tmp cand dir...",output_file=cutterfile)
        os.system("rm "+remote_cand_dir + "/" + cand_isot + "*"+suff+"*")
        printlog("done",output_file=cutterfile)
    printlog("Done! Total Remaining Candidates: " + str(len(finalidxs)),output_file=cutterfile)
    return


def submit_cand_nsfrb(image,searched_image,TOAs,fname,uv_diag,dec_obs,args,suff,tsamp_use,DM_trials_use,cand_isot,cand_mjd,RA_axis,DEC_axis,RA_axis_2D,DEC_axis_2D,nsamps,injection_flag,postinjection_flag,slow,imgdiff,client,PSF,ffalock,plotlock):
    """
    Modelled from dsa110-T3/dsaT3/T3_manager.submit_cand(); Given filename of trigger json,
    create DSACand and submit to scheduler for T3 processing
    """


    #(1) sort candidates
    canddict = dict()
    d_sort = client.submit(cc.sort_cands,fname,searched_image,TOAs,args.SNRthresh,RA_axis,DEC_axis,widthtrials,DM_trials_use,canddict,
                            np.abs(image.shape[1]-searched_image.shape[1]),
                            np.abs(image.shape[0]-searched_image.shape[0]),
                            cutterfile,0,args.maxcands,args.writeraw,args.completeness,False,args.completeness,args.searchradius)#,lock=lock,priority=1,resources={'MEMORY': 10e9})

    #(2) clustering
    d_cluster = client.submit(cluster_manage,d_sort,image,nsamps,dec_obs,args,cutterfile,DM_trials_use,widthtrials,injection_flag,postinjection_flag,PSF)#,lock=lock,priority=1,resources={'MEMORY': 10e9})

    #(3) classifying
    d_classify = client.submit(classify_manage,d_cluster,image,nsamps,nchans,dec_obs,args,cutterfile,DM_trials_use,widthtrials,cand_isot,injection_flag,postinjection_flag,slow,imgdiff)#,lock=lock,priority=1,resources={'MEMORY': 10e9})

    #(3.5) fast folding
    d_ffa = client.submit(ffa_manage,d_classify,image,nsamps,nchans,dec_obs,args,cutterfile,DM_trials_use,widthtrials,cand_isot,injection_flag,postinjection_flag,slow,imgdiff,RA_axis_2D,DEC_axis_2D,tsamp_ms,ffalock,suff)

    #(4) writing csvs and jsons for remaining cands (might not be any remaining cands)
    d_write = client.submit(writecands_manage,d_ffa,image,args,DM_trials_use,widthtrials,suff,cand_isot,cand_mjd,slow,imgdiff,injection_flag,postinjection_flag,tsamp_use,nsamps,RA_axis_2D,DEC_axis_2D,cutterfile)#,lock=lock,priority=1,resources={'MEMORY': 10e9})

    #(5) sending alerts (slack, triggers, etc)
    d_trigger = client.submit(sendtrigger_manage,d_write,image,searched_image,args,uv_diag,dec_obs,slow,imgdiff,RA_axis,DEC_axis,DM_trials_use,widthtrials,cand_isot,suff,cutterfile,injection_flag,postinjection_flag,plotlock)#,lock=lock,priority=1,resources={'MEMORY': 10e9})
    
    #(6) archiving
    d_archive = client.submit(archive_manage,d_trigger,cand_isot,suff,cutterfile,injection_flag,postinjection_flag)#,lock=lock,priority=1,resources={'MEMORY': 10e9})

    if args.trigger:
        d_cs = client.submit(run_createstructure_nsfrb, d_trigger, args.daskaddress, key=f"run_createstructure_nsfrb-{d.trigname}")#, lock=lock, priority=1)  # create directory structure
        d_vc = client.submit(run_voltagecopy_nsfrb, d_cs, args.daskaddress, key=f"run_voltagecopy_nsfrb-{d.trigname}")#, lock=lock)  # copy voltages
        d_h5 = client.submit(run_hdf5copy_nsfrb, d_cs, args.daskaddress, key=f"run_hdf5copy_nsfrb-{d.trigname}")#, lock=lock)  # copy hdf5
        fut = client.submit(run_final_nsfrb, (d_h5, d_cs, d_vc), args.daskaddress, key=f"run_final_nsfrb-{d.trigname}")#, lock=lock)
        return fut
    return d_archive

def run_createstructure_nsfrb(d_future, daskaddress, lock=None):
    """ Use DSACand (after filplot) to decide on creating/copying files to candidate data area.
    """
    if len(daskaddress)>0:
        res = d_future
    else:
        res = d_future.result()
    if res is None:
        return
    d = res[-1]

    if d.real and not d.injected:
        print("Running createstructure for real/non-injection candidate.")

        # TODO: have DataManager parse DSACand
        dm = data_manager.NSFRBDataManager(d.__dict__)
        # TODO: have update method accept dict or DSACand
        d.update(dm())

    else:
        print("Not running createstructure for non-astrophysical candidate.")

    d.writejson(outpath=OUTPUT_PATH, lock=lock)
    return d


def run_burstfit(d, lock=None):
    """ Given DSACand, run burstfit analysis.
    Returns new dictionary with refined DM, width, arrival time.
    """

    from burstfit.BurstFit_paper_template import real_time_burstfit

    if d.real:
        print('Running burstfit on {0}'.format(d.trigname))
        LOGGER.info('Running burstfit on {0}'.format(d.trigname))
        d_bf = real_time_burstfit(d.trigname, d.filfile, d.snr, d.dm, d.ibox)

        d.update(d_bf)
        d.writejson(outpath=OUTPUT_PATH, lock=lock)
    else:
        print('Not running burstfit on {0}'.format(d.trigname))
        LOGGER.info('Not running burstfit on {0}'.format(d.trigname))

    return d


def run_hdf5copy_nsfrb(d_future, daskaddress, lock=None):
    """ Given DSACand (after filplot), copy hdf5 files
    """
    if len(daskaddress)>0:
        res = d_future
    else:
        res = d_future.result()
    if res is None:
        return
    d = d_future

    if d.real and not d.injected:
        print('Running hdf5copy on {0}'.format(d.trigname))
        LOGGER.info('Running hdf5copy on {0}'.format(d.trigname))

        dm = data_manager.DataManager(d.__dict__)
        dm.link_hdf5_files()

        d.update(dm.candparams)
        d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d

def run_voltagecopy_nsfrb(d_future, daskaddress, lock=None):
    """ Given DSACand (after filplot), copy voltage files.
    """
    if len(daskaddress)>0:
        res = d_future
    else:
        res = d_future.result()
    if res is None:
        return
    d = d_future

    if d.real and not d.injected:
        print('Running voltagecopy on {0}'.format(d.trigname))
        #LOGGER.info('Running voltagecopy on {0}'.format(d.trigname))
        dm = data_manager.NSFRBDataManager(d.__dict__)
        dm.copy_voltages()

        # TODO: have update method accept dict or DSACand
        d.update(dm.candparams)
        d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d

def run_hires(ds, lock=None):
    """ Given DSACand objects from burstfit and voltage, generate hires filterbank files.
    """

    d, d_vc = ds
    d.update(d_vc)

    print('placeholder run_hires on {0}'.format(d.trigname))
    #LOGGER.info('placeholder run_hires on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_pol(d, lock=None):
    """ Given DSACand (after hires), run polarization analysis.
    Returns updated DSACand with new file locations?
    """

    print('placeholder nrun_pol on {0}'.format(d.trigname))
    #LOGGER.info('placeholder run_pol on {0}'.format(d.trigname))

#    if d_hr['real'] and not d_hr['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_fieldmscopy(d, lock=None):
    """ Given DSACand (after filplot), copy field MS file.
    Returns updated DSACand with new file locations.
    """

    print('placeholder run_fieldmscopy on {0}'.format(d.trigname))
    #LOGGER.info('placeholder run_fieldmscopy on {0}'.format(d.trigname))

#    if d_fp['real'] and not d_fp['injected']:
#        dm = data_manager.DataManager(d_fp)
#        dm.link_field_ms()
#        update_json(dm.candparams, lock=lock)
#        return dm.candparams
#    else:
    return d


def run_candidatems(ds, lock=None):
    """ Given DSACands from filplot and voltage copy, make candidate MS image.
    Returns updated DSACand with new file locations.
    """

    d, d_vc = ds
    d.update(d_vc)

    print('placeholder run_candidatems on {0}'.format(d.trigname))
    #LOGGER.info('placeholder run_candidatems on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_hiresburstfit(d, lock=None):
    """ Given DSACand, run highres burstfit analysis.
    Returns updated DSACand with new file locations.
    """

    print('placeholder run_hiresburstfit on {0}'.format(d.trigname))
    #LOGGER.info('placeholder run_hiresburstfit on {0}'.format(d.trigname))

#    if d_hr['real'] and not d_hr['injected']:
    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_imloc(d, lock=None):
    """ Given DSACand (after candidate image MS), run image localization.
    """

    print(f'Running localization on {d.trigname}')
    #LOGGER.info(f'Running localization on {d.trigname}')

# TODO: is this the first sent or an update with good position?
#    if d.real and not d.injected:
#        dc.set('observation', args=asdict(d))

    d.writejson(outpath=OUTPUT_PATH, lock=lock)
    return d


def run_astrometry(ds, lock=None):
    """ Given field image MS and candidate image MS, run astrometric localization analysis.
    """

    d, d_cm = ds
    d.update(d_cm)

    print('placeholder run_astrometry on {0}'.format(d.trigname))
    #LOGGER.info('placeholder run_astrometry on {0}'.format(d.trigname))

#    if dd['real'] and not dd['injected']:

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def run_final_nsfrb(ds, daskaddress, lock=None):
    """ Reduction task to handle all final tasks in graph.
    May also update etcd to notify of completion.
    """

#    d, d_po, d_hb, d_il, d_as = ds
    
    if len(daskaddress)>0:
        d, d_fm, d_vc = ds
    else:
        d, d_fm, d_vc = [ds[i].result() for i in range(len(ds))]

    d.update(d_fm)
    d.update(d_vc)
#    d.update(d_il)
#    d.update(d_as)

    print('Final merge of results for {0}'.format(d.trigname))
    #LOGGER.info('Final merge of results for {0}'.format(d.trigname))

    d.writejson(outpath=OUTPUT_PATH, lock=lock)

    return d


def wait_for_local_file(fl, timeout, allbeams=False):
    """ Wait for file named fl to be written. fl can be string filename of list of filenames.
    If timeout (in seconds) exceeded, then return None.
    allbeams will parse input (str) file name to get list of all beam file names.
    """

    if allbeams:
        assert isinstance(fl, str), 'Input should be detection beam fil file'
        loc = os.path.dirname(fl)
        fl0 = os.path.basename(fl.rstrip('.fil'))
        fl1 = "_".join(fl0.split("_")[:-1])
        fl = [f"{os.path.join(loc, fl1 + '_' + str(i) + '.fil')}" for i in range(512)]

    if isinstance(fl, str):
        fl = [fl]
    assert isinstance(fl, list), "name or list of fil files expected"

    elapsed = 0
    while not all([os.path.exists(ff) for ff in fl]):
        time.sleep(5)
        elapsed += 5
        if elapsed > timeout:
            return None
        elif elapsed <= 5:
            print(f"Waiting for {len(fl)} files, like {fl[0]}...")

    return fl




"""
ETCD AND QUEUE
"""

from multiprocessing import Process, Queue
import dsautils.dsa_store as ds
ETCD = ds.DsaStore()
ETCDKEY = f'/mon/nsfrb/candidates'
QQUEUE = Queue()
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,pixperFWHM,nchans,remote_cand_dir

def etcd_to_queue(etcd_dict,queue=QQUEUE):
    """
    This is a callback function that takes a candidate from etcd and adds it to the cand cutter queue
    """
    printlog("found etcd candidate:" ,output_file=cutterfile)
    printlog(etcd_dict,output_file=cutterfile)
    printlog("putting in queue",output_file=cutterfile)
    queue.put(etcd_dict['candfile'])
    queue.put(etcd_dict['uv_diag'])
    queue.put(etcd_dict['dec'])
    queue.put(etcd_dict['img_shape'])
    queue.put(etcd_dict['img_search_shape'])
    return

#from realtime.rtreader import rtread_cand
from nsfrb.config import NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,NSFRB_CANDDADA_SLOW_KEY,NSFRB_SRCHDADA_SLOW_KEY,NSFRB_TOADADA_SLOW_KEY,NSFRB_CANDDADA_IMGDIFF_KEY,NSFRB_SRCHDADA_IMGDIFF_KEY,NSFRB_TOADADA_IMGDIFF_KEY
def main(args):
    if len(args.daskaddress)==0:
        ffalock_ = Lock()
        plotlock_ = Lock()
    else:
        ffalock_ = Lock_DASK()
        plotlock_ = Lock_DASK()
    #sys.stderr = open(error_file,"w")
    printlog("Starting T4 Manager (realtime candcutter)...",output_file=cutterfile)
    printlog("Adding ETCD watch on key "+ETCDKEY,output_file=cutterfile)
    if not args.testtrigger:
        ETCD.add_watch(ETCDKEY, etcd_to_queue)
    tasklist=[]
    tasktimes=[]
    counter = 0
    if len(args.daskaddress)==0:
        client = ThreadPoolExecutor(args.maxProcesses)
    else:
        client = Client(args.daskaddress)


    #get system parameters at runtime
    test, key_string, nant, nchan, npol, fobs, samples_per_frame, samples_per_frame_out, nint, nfreq_int, antenna_order, pt_dec, tsamp, fringestop, filelength_minutes, outrigger_delays, refmjd, subband = pu.parse_params(param_file=None)
    printlog("System declination:" + str(pt_dec*180/np.pi),output_file=cutterfile)

    if args.psfcluster:
        PSF,PSF_params = scPSF.manage_PSF(scPSF.make_PSF_dict(),(2*args.gridsize)+1,pt_dec*180/np.pi,nsamps=init_nsamps)#scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),image.shape[0],True,nsamps).mean((2,3))
        PSF = PSF.mean((2,3))
        printlog("PSF shape for clustering:" + str(PSF.shape),output_file=cutterfile)
    else:
        PSF = None

    while True:
        if not args.testtrigger:
            printlog("Looking for cands in queue:" + str(QQUEUE),output_file=cutterfile)        
            fname = (remote_cand_dir if args.remote else raw_cand_dir) + str(QQUEUE.get())
            uv_diag = float(QQUEUE.get())#np.frombuffer(bytes.fromhex(QQUEUE.get()))[0]
            dec_obs = float(QQUEUE.get())#np.frombuffer(bytes.fromhex(QQUEUE.get()))[0]
            img_shape = tuple(QQUEUE.get())
            img_search_shape = tuple(QQUEUE.get())
            #assert(np.abs((dec_obs*np.pi/180) - pt_dec)<1e-2)
            if np.abs((dec_obs*np.pi/180) - pt_dec)>1e-2:
                pt_dec = (dec_obs*np.pi/180)
                if args.psfcluster:
                    PSF,PSF_params = scPSF.manage_PSF(scPSF.make_PSF_dict(),(2*args.gridsize)+1,pt_dec*180/np.pi,nsamps=init_nsamps)#scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),image.shape[0],True,nsamps).mean((2,3))
                    PSF = PSF.mean((2,3))
                    printlog("PSF shape for clustering:" + str(PSF.shape),output_file=cutterfile)
                else:
                    PSF = None
            printlog("Cand Cutter found cand file " + str(fname),output_file=cutterfile)
        else:
            cand_isot = "2025-06-10T12:05:00.000"#Time(Time.now().mjd - (50/60/24),format='mjd').isot
            fname = (remote_cand_dir if args.remote else raw_cand_dir) + "candidates_" + cand_isot + ".csv"
            uv_diag = 500
            dec_obs = 71.6
            img_shape = (301,301)
            img_search_shape = (301,301)
            printlog("Generating injected cand")
        slow = 'slow' in fname
        imgdiff = 'imgdiff' in fname
        if slow:
            suff = '_slow'
            tsamp_use = tsamp_slow
            DM_trials_use = DM_trials_slow
            printlog("SLOW CANDCUTTING",output_file=cutterfile)
        elif imgdiff:
            suff = '_imgdiff'
            tsamp_use = tsamp_imgdiff
            DM_trials_use = DM_trials
            printlog("IMGDIFF CANDCUTTING",output_file=cutterfile)
        else:
            suff = ""
            tsamp_use = tsamp_ms
            DM_trials_use = DM_trials
        cand_isot = fname[fname.index("candidates_")+11:fname.index(suff + ".csv")]
        
        """
        if 'slow' in fname:
            rtkey1 = NSFRB_CANDDADA_SLOW_KEY
            rtkey2 = NSFRB_SRCHDADA_SLOW_KEY
            rtkey3 = NSFRB_TOADADA_SLOW_KEY
        elif 'imgdiff' in fname:
            rtkey1 = NSFRB_CANDDADA_IMGDIFF_KEY
            rtkey2 = NSFRB_SRCHDADA_IMGDIFF_KEY
            rtkey3 = NSFRB_TOADADA_IMGDIFF_KEY
        else:
            rtkey1 = NSFRB_CANDDADA_KEY
            rtkey2 = NSFRB_SRCHDADA_KEY
            rtkey3 = NSFRB_TOADADA_KEY
        image = rtread_cand(key=rtkey1,gridsize_dec=img_shape[0],gridsize_ra=img_shape[1],nsamps=img_shape[2],nchans=img_shape[3])
        searched_image = rtread_cand(key=rtkey2,gridsize_dec=img_search_shape[0],gridsize_ra=img_search_shape[1],nsamps=img_search_shape[2],nchans=img_search_shape[3])
        TOAs = (rtread_cand(key=rtkey3,gridsize_dec=img_search_shape[0],gridsize_ra=img_search_shape[1],nsamps=img_search_shape[2],nchans=img_search_shape[3])).astype(int)
        """
        if args.remote:
            #update injections
            os.system("scp h24.pro.pvt:/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-injections/injections.csv " + inject_dir)
        if not args.testtrigger:
            try:
                #if remote, copy from h24
                if args.remote:
                    os.system("scp h24.pro.pvt:"+raw_cand_dir + cand_isot + suff + ".npy "+remote_cand_dir)
                    os.system("scp h24.pro.pvt:"+raw_cand_dir + cand_isot + suff + "_searched.npy "+remote_cand_dir)
                    os.system("scp h24.pro.pvt:"+raw_cand_dir + cand_isot + suff + "_TOAs.npy "+remote_cand_dir)
                    image = np.load(remote_cand_dir + cand_isot + suff + ".npy")
                    searched_image = np.load(remote_cand_dir+ cand_isot + suff + "_searched.npy")
                    TOAs = np.load(remote_cand_dir+ cand_isot + suff + "_TOAs.npy").astype(int)
                else:
                    image = np.load(raw_cand_dir + cand_isot + suff + ".npy")
                    searched_image = np.load(raw_cand_dir + cand_isot + suff + "_searched.npy")
                    TOAs = np.load(raw_cand_dir + cand_isot + suff + "_TOAs.npy").astype(int)
            except Exception as e:
                printlog("No image found for candidate " + cand_isot,output_file=cutterfile)
                printlog(str(e),output_file=cutterfile)
                if not args.remote:
                    os.system("rm " +  raw_cand_dir + "*" + cand_isot + "*"+suff+"*")
                #return
                continue

        else:
            from scipy.stats import uniform
            searched_image = uniform.rvs(loc=0,scale=args.SNRthresh+1,size=((301,301,5,16)))
            image = uniform.rvs(size=((301,301,25,16)))
            TOAs = np.zeros((301,301,25,16),dtype=int)
        cand_mjd = Time(cand_isot,format='isot').mjd
        injection_flag,postinjection_flag = cc.is_injection(cand_isot)
        injection_flag = injection_flag and (not args.completeness)
        postinjection_flag = postinjection_flag and (not args.completeness)
        RA_axis,DEC_axis,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs,pixperFWHM=pixperFWHM)
        RA_axis = RA_axis[-searched_image.shape[1]:]
        RA_axis_2D,DEC_axis_2D,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs,two_dim=True,pixperFWHM=pixperFWHM)
        RA_axis_2D = RA_axis_2D[:,-searched_image.shape[1]:]
        DEC_axis_2D = DEC_axis_2D[:,-searched_image.shape[1]:]
        nsamps = image.shape[2]

        


        #submit task
        #tasktimes.append(time.time())
        tasklist.append(submit_cand_nsfrb(image,searched_image,TOAs,fname,uv_diag,dec_obs,args,suff,tsamp_use,DM_trials_use,cand_isot,cand_mjd,
                        RA_axis,DEC_axis,RA_axis_2D,DEC_axis_2D,nsamps,injection_flag,postinjection_flag,slow,imgdiff,client,PSF,ffalock_,plotlock_)) 
        """
        poplist = []
        for ti in range(len(tasklist)):
            if (not tasklist[ti].done()) and (time.time()-tasktimes[ti] >= (args.tasktimeout*60)):
                r = tasklist[ti].result()
                poplist.append(ti)
        for p in poplist:
            tasklist.pop(p)
            tasktimes.pop(p)
        """

        #client.submit(cc.candcutter_task,fname,uv_diag,dec,img_shape,img_search_shape,vars(args),resources={'MEMORY':10e9})
        if args.sleep > 0:
            printlog("Sleeping for " + str(args.sleep/60) + " minutes",output_file=cutterfile)
            time.sleep(args.sleep)
        elif args.testtrigger:
            for i in range(len(tasklist)):
                res = tasklist[i].result()
                print(res)
            printlog("Sleeping for " + str(60/60) + " minutes",output_file=cutterfile)
            time.sleep(600)

    return 0

if __name__=="__main__":
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutout', action='store_true', help='Get image cutouts around each candidate')
    parser.add_argument('--subimgpix',type=int,help='Length of image cutouts in pixels, default=11',default=11)
    parser.add_argument('--cluster',action='store_true',help='Enable clustering with HDBSCAN')
    parser.add_argument('--plotclusters',action='store_true',help='Plot intermediate plots from HDBSCAN clustering')
    parser.add_argument('--mincluster',type=int,help='Minimum number of candidates required to be made a separate HDBSCAN cluster,default=5',default=5)
    parser.add_argument('--minsamples',type=int,help='Minimum number of candidates to be core point,default=2',default=2)
    parser.add_argument('--verbose',action='store_true', help='Enable verbose output')
    parser.add_argument('--classify',action='store_true', help='Classify candidates with a machine learning convolutional neural network')
    parser.add_argument('--classify3D',action='store_true', help='Classify candidates with a machine learning convolutional neural network with time dependence')
    parser.add_argument('--classcut',action='store_true',help='Only save candidates that the classifier passes')
    parser.add_argument('--model_weights', type=str, help='Path to the model weights file',default=cwd + "/simulations_and_classifications/model_weights_20250212.pth")
    parser.add_argument('--model_weights3D',type=str, help='Path to the model weights file for 3D classifying',default=cwd + "/simulations_and_classifications/enhanced3dcnn_weights_final_remote.pth")
    parser.add_argument('--toslack',action='store_true',help='Sends Candidate Summary Plots to Slack')
    parser.add_argument('--sleep',type=float,help='Time in seconds to sleep between successive cand_cutter runs; default=0',default=0)
    parser.add_argument('--runtime',type=float,help='Minimum time in seconds to run before sleep cycle; default=60',default=60)
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of threads for thread pool; default=5',default=5)
    parser.add_argument('--archive',action='store_true',help='Archive candidates on dsastorage')
    parser.add_argument('--maxcands',type=int,help='Maximum number of candidates searchable in one iteration. Default is full image, 300x300x5x16=7.2e6',default=int(7.2e6 +1))
    parser.add_argument('--maxcands_postcluster',type=int,help='Maximum number of candidates post clustering. Default is full image, 300x300x5x16=7.2e6',default=int(7.2e6 +1))
    parser.add_argument('--percentile',type=float,help='Percentile above which to take candidates, e.g. if 90, candidates with s/n in 90th percentile will be clustered. Default 0',default=0.0)
    parser.add_argument('--psfpercentile',type=float,help='Percentile to use for PSF clustering, default 70th',default=70.0)
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 10',default=10)
    parser.add_argument('--train',action='store_true',help='Save candidate cutouts to the training set for the ML classifier')
    parser.add_argument('--traininject',action='store_true',help='Save injection cutouts to the training set for the ML classifier')
    parser.add_argument('--trigger',action='store_true',help='Send T4 trigger to copy visibility buffer and voltages for each candidate event')
    parser.add_argument('--useTOA',action='store_true',help='Include TOAs in clustering algorithm')
    parser.add_argument('--psfcluster',action='store_true',help='PSF-based spatial clustering')
    parser.add_argument('--clusteriters',type=int,help='Number of clustering iterations; minimum cluster size reduced on each iteration; default=1',default=1)
    parser.add_argument('--maxcand',action='store_true',help='If set, takes only the maximum S/N candidate in each chunk after clustering; otherwise returns all candiddates above S/N threshold')
    parser.add_argument('--pixperFWHM',type=float,help='Pixels per FWHM, default 3',default=pixperFWHM)
    parser.add_argument('--avgcluster',action='store_true', help='Average parameters of each cluster; if not set, takes peak cluster member parameters')
    parser.add_argument('--writeraw',action='store_true',help='Write raw candidates to a csv file')
    parser.add_argument('--testtrigger',action='store_true',help='Inject fake data to test T4')
    parser.add_argument('--daskaddress',type=str,help='Address for dask scheduler',default="")
    parser.add_argument('--gridsize',type=int,help='Expected length in pixels for each sub-band image, SHOULD ALWAYS BE ODD, default='+str(IMAGE_SIZE),default=IMAGE_SIZE)
    parser.add_argument('--GP',action='store_true',help='notifies T4 manager that data from the galactic plane survey is being searched')
    parser.add_argument('--periods',nargs='+',type=int,help='periods (in samples) to search',default=[5,10,15,30,45])
    parser.add_argument('--FFA',action='store_true',help='run fast-folding periodicity search on any single pulse candidates')
    parser.add_argument('--FFAgulps',type=int,help='Number of gulps for ffa search',default=1)
    parser.add_argument('--FFAbin',type=int,help='Downsampling factor for ffa search',default=1)
    parser.add_argument('--FFASNRthresh',type=float,help='S/N threshold for periodicity search',default=3)
    parser.add_argument('--tasktimeout',type=float,help='Max time to allow task to run in background before forcing to foreground,default=1 minute',default=1)
    parser.add_argument('--FFAbinchans',action='store_true',help='Average over all channels in each sub-band for periodicity search')
    parser.add_argument('--completeness',action='store_true',help='Run a completeness assessment by sending images to the process server and testing recovery')
    parser.add_argument('--searchradius',type=float,help='Max search radius in degrees within which to include candidates,default=inf',default=np.inf)
    parser.add_argument('--remote',action='store_true',help='Run T4 manager on remote server; files are scp to/from h24')
    args = parser.parse_args()
    
    main(args)

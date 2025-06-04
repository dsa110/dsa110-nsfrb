from nsfrb.config import tsamp,tsamp_slow,fmin,fmax,nchans,nsamps,NUM_CHANNELS, CH0, CH_WIDTH, AVERAGING_FACTOR, IMAGE_SIZE, c, Lon,Lat, DM_tol,table_dir,tsamp_imgdiff
from nsfrb.imaging import uv_to_pix
from nsfrb.searching import gen_dm_shifts,widthtrials,DM_trials,DM_trials_slow,gen_boxcar_filter,default_PSF
import argparse
from dsautils import dsa_store
from nsfrb import candcutting as cc
import json
import os
import sys
import numpy as np
from astropy.time import Time
from nsfrb.config import tsamp,baseband_tsamp,tsamp_slow,tsamp_imgdiff
from nsfrb.config import nsamps as init_nsamps
from nsfrb.outputlogging import printlog
from nsfrb import candcutting
from event import event
from dsaT4 import data_manager
from dask.distributed import Client, Lock
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
client = Client('tcp://10.42.0.228:8786')#10.42.0.232:8786')
LOCK = Lock('update_json')
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

final_cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/final_cands/candidates/"
def nsfrb_to_json(cand_isot,mjds,snr,width,dm,ra,dec,trigname,final_cand_dir=final_cand_dir,slow=False,imgdiff=False):
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
        ibox = int(np.ceil(width*tsamp/baseband_tsamp))
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
               "trigname":trigname},f)
    f.close()

    return final_cand_dir + "/" + trigname + ".json"


LOCK = Lock('update_json')
from nsfrb.config import cutterfile
from simulations_and_classifications import generate_PSF_images as scPSF
def cluster_manage(d_future,image,nsamps,dec_obs,args,cutterfile,DM_trials_use,widthtrials,injection_flag,postinjection_flag):
    raw_cand_names,finalcands = d_future.result()
    if not args.cluster or len(finalcands)>=args.mincluster:
        return raw_cand_names,finalcands
    
    finalidxs = np.arange(len(finalcands),dtype=int)
    useTOA=args.useTOA and len(finalcands[0])==6
    #start clustering
    printlog("clustering with HDBSCAN...",output_file=cutterfile)
    #clustering with hdbscan
    if args.psfcluster:
        PSF,PSF_params = scPSF.manage_PSF(scPSF.make_PSF_dict(),(2*image.shape[0])+1,dec_obs,nsamps=nsamps)#scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),image.shape[0],True,nsamps).mean((2,3))
        PSF = PSF.mean((2,3))
        printlog("PSF shape for clustering:" + str(PSF.shape),output_file=cutterfile)
    else:
        PSF = None
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


def classify_manage(d_future,image,nsamps,nchans,dec_obs,args,cutterfile,DM_trials_use,widthtrials,cand_isot,injection_flag,postinjection_flag):
    finalcands,finalidxs = d_future.result()
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
                subimg = get_subimage(image,int(finalcands[j][0]),int(finalcands[j][1]),save=False,subimgpix=args.subimgpix)
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
                data_array[j,:,:,:,:] = get_subimage(image,int(finalcands[j][0]),int(finalcands[j][1]),save=False,subimgpix=args.subimgpix)
                printlog("cand shape:" + str(data_array[j,:,:,:].shape),output_file=cutterfile)

        #run classifier
        printlog("still fine",output_file=cutterfile)
        printlog("Start classifying " + str(data_array.shape),output_file=cutterfile)
        predictions, probabilities = classify_images_3D(data_array, args.model_weights3D, verbose=args.verbose)
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
            return
    return finalcands,finalidxs,predictions,probabilities

from nsfrb.searching import maxshift
def writecands_manage(d_future,image,args,DM_trials_use,widthtrials,suff,cand_isot,slow,imgdiff,injection_flag,postinjection_flag):
    res = d_future.result()
    if res is None:
        return
    elif len(res) == 4:
        finalcands,finalidxs,predictions,probabilities = res
        classify_flag = True
    elif len(res) == 2:
        finalcands,finalidxs = res
        classify_flag = False
    else:
        return
    
    #if its an injection write the highest SNR candidate to the injection tracker
    useTOA=args.useTOA and len(finalcands[0])==6
    #injection_flag,postinjection_flag = cc.is_injection(cand_isot)
    if injection_flag:
        with open(recover_file,"a") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            for j in finalidxs:
                wr.writerow([cand_isot,DM_trials_use[int(finalcands[j][3])],widthtrials[int(finalcands[j][2])],finalcands[j][-1],(None if not classify_flag else predictions[j]),(None if not classify_flag else probabilities[j])])
        csvfile.close()


    #make final directory for candidates
    os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff)

    #write final candidates to csv
    prefix = "NSFRB"
    with open(table_dir+"nsfrb_lastname.txt","r") as lnamefile:
        lastname = (lnamefile.read()).strip()
        if lastname == "None":
            lastname = None
    lnamefile.close()

    #lastname =      #once we have etcd, change to 'names.get_lastname()'
    allcandnames = []
    csvfile = open(final_cand_dir+ str("injections" if injection_flag else "candidates")  + "/" + cand_isot + suff + "/final_candidates_" + cand_isot + ".csv","w")
    wr = csv.writer(csvfile,delimiter=',')
    hdr = ["candname","RA index","DEC index","WIDTH index", "DM index"]
    if useTOA: hdr += ["TOA"]
    hdr += ["SNR"]
    if classify_flag: hdr += ["PROB"]
    wr.writerow(hdr)
    sysstdout = sys.stdout
    for j in finalidxs:#range(len(finalidxs)):
        with open(cutterfile,"a") as sys.stdout:
            lastname = names.increment_name(cand_mjd,lastname=lastname)
        sys.stdout = sysstdout
        if classify_flag:
            wr.writerow(np.concatenate([[lastname],np.array(finalcands[j][:-1],dtype=int),[finalcands[j][-1]],[probabilities[j]]]))
        else:
            wr.writerow(np.concatenate([[lastname],np.array(finalcands[j][:-1],dtype=int),[finalcands[j][-1]]]))
        allcandnames.append(prefix + lastname)
    csvfile.close()

    with open(table_dir+"nsfrb_lastname.txt","w") as lnamefile:
        if lastname is not None:
            lnamefile.write(lastname)
        else:
            lnamefile.write("None")
    lnamefile.close()
    printlog("done naming stuff",output_file=cutterfile)

    #make subdirectories for candidates
    for j in finalidxs:

        lastname = allcandnames[j]
        #make folder for each candidate
        os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname)
        os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname + "/voltages")

    #send candidates to slack 
    if len(finalidxs) > 0:
        #make diagnostic plot
        printlog("making diagnostic plot...",output_file=cutterfile,end='')
        canddict = dict()
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
        sourceimg_all = np.concatenate([np.zeros(tuple(list(image.shape[:2])+[0 if (slow or imgdiff) else maxshift]+[image.shape[3]])),image],axis=2)
        for i in range(len(finalidxs)):
            DM = DM_trials_use[int(canddict['dm_idxs'][i])]

            sourceimg = sourceimg_all[int(canddict['dec_idxs'][i]):int(canddict['dec_idxs'][i])+1,
                                    int(canddict['ra_idxs'][i]):int(canddict['ra_idxs'][i])+1,:,:]#np.concatenate([np.zeros((1,1,maxshift,image.shape[3])),image[canddict['dec_idxs'][i],canddict['ra_idxs'][i],:,:],axis=2)
            if (DM != 0 and not imgdiff):
                printlog("COMPUTING SHIFTS FOR DM="+str(DM)+"pc/cc "+ str(sourceimg.shape),output_file=cutterfile)
                corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append,wraps_append,wraps_no_append = gen_dm_shifts(np.array([DM]),freq_axis,tsamp_use,nsamps,outputwraps=True,maxshift=0 if (slow or imgdiff) else maxshift)

                printlog("corr shifts shape:" + str(corr_shifts_all_append.shape),output_file=cutterfile)

                DM_idx = 0#list(DM_trials).index(DM)
                printlog("PRE-DM SHAPE:"+str(sourceimg.shape),output_file=cutterfile)
                sourceimg_dm = (((((np.take_along_axis(sourceimg[:,:,:,np.newaxis,:].repeat(1,axis=3).repeat(2,axis=4),indices=corr_shifts_all_append[:,:,:,DM_idx:DM_idx+1,:],axis=2))*tdelays_frac_append[:,:,:,DM_idx:DM_idx+1,:]))[:,:,:,0,:]))
                printlog("POST-DM SHAPE:"+str(sourceimg_dm.shape),output_file=cutterfile)
                #zero out anywhere that was wrapped
                #sourceimg_dm[wraps_no_append[:,:,:,DM_idx,:].repeat(sourceimg.shape[0],axis=0).repeat(sourceimg.shape[1],axis=1)] = 0

                #now average the low and high shifts 
                sourceimg_dm = (sourceimg_dm.reshape(tuple(list(sourceimg.shape)[:2] + [nsamps,nchans] + [2])).sum(4))

            else:
                sourceimg_dm = sourceimg
            timeseries.append(np.nanmean(sourceimg_dm,(0,1,3)))

            if not injection_flag:
                #create json file
                snr=canddict['snrs'][i]
                width=int(widthtrials[int(canddict['wid_idxs'][i])])
                dm=int(DM_trials_use[int(canddict['dm_idxs'][i])])
                ra=RA_axis_2D[int(canddict['dec_idxs'][i]),int(canddict['ra_idxs'][i])] #RA_axis[int(canddict['ra_idxs'][i])]
                dec=DEC_axis_2D[int(canddict['dec_idxs'][i]),int(canddict['ra_idxs'][i])] #DEC_axis[int(canddict['dec_idxs'][i])]
                trigname = canddict['names'][i]
                printlog(str(snr) +","+ str(width)+","+str(dm) + ","+ str(ra) + "," + str(dec) + "," + trigname,output_file=cutterfile)
                if useTOA:
                    toa = canddict['TOAs'][i]
                    cand_mjd = Time(Time(cand_isot,format='isot').mjd + (canddict['TOAs'][i]*(tsamp_use)/1000/86400),format='mjd').mjd
                else:
                    cand_mjd = Time(cand_isot,format='isot').mjd
                
                fl = nsfrb_to_json(cand_isot,cand_mjd,snr,width,dm,ra,dec,trigname,final_cand_dir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + trigname + "/",slow=slow,imgdiff=imgdiff)
                printlog(fl,output_file=cutterfile)
    if len(res) == 4:
        if not injection_flag:
            return finalcands,finalidxs,predictions,probabilities,canddict,allcandnames,fl
        else:
            return finalcands,finalidxs,predictions,probabilities,canddict,allcandnames
    elif len(res) == 2:
        if not injection_flag:
            return finalcands,finalidxs,canddict,allcandnames,fl
        else:
            return finalcands,finalidxs,canddict,allcandnames
    return

def sendtrigger_manage(d_future,image,searched_image,args,uv_diag,dec_obs,slow,imgdiff,RA_axis,DEC_axis,DM_trials_use,widthtrials,output_dir,cand_isot,suff,cutterfile,injection_flag,postinjection_flag):
    res = d_future.result()
    if res is None:
        return
    elif len(res) == 7:
        finalcands,finalidxs,predictions,probabilities,canddict,allcandnames,jsonfname = res
        classify_flag = True
        #injection_flag = True
    elif len(res) == 6:
        finalcands,finalidxs,predictions,probabilities,allcandnames,canddict = res
        classify_flag = True
        #injection_flag = False
    elif len(res) == 5:
        finalcands,finalidxs,canddict,allcandnames,jsonfname = res
        classify_flag = False
        #injection_flag = True
    elif len(res) == 4:
        finalcands,finalidxs,allcandnames,canddict = res
        classify_flag = False
        #injection_flag = False
    else:
        return

    printlog("Creating candplot...",output_file=cutterfile)
    candplot=pl.search_plots_new(canddict,image,cand_isot,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                            DM_trials=DM_trials_use,widthtrials=widthtrials,
                                            output_dir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/",show=False,s100=args.SNRthresh/2,
                                            injection=injection_flag,vmax=args.SNRthresh+2,vmin=args.SNRthresh,
                                            searched_image=searched_image,timeseries=timeseries,uv_diag=uv_diag,
                                            dec_obs=dec_obs,slow=slow,imgdiff=imgdiff)

    if args.toslack:
        printlog("sending plot to slack...",output_file=cutterfile)
        send_candidate_slack(candplot,filedir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/")

    if args.trigger:
        T4trigger = event.create_event(fl)
        return list(res) + [candplot, T4trigger]
    else:
        return list(res) + [candplot]

def archive_manage(d_future,cand_isot,suff,cutterfile,injection_flag,postinjection_flag):
    if not args.archive or 'NSFRBT4' not in os.environ.keys():
        return None

    res = d_future.result()
    if res is None:
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
        printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname + "/*" + " user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/" + lastname + "/",output_file=cutterfile)
        os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname + "/*" + " user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/" + lastname + "/")

    printlog("Done! Total Remaining Candidates: " + str(len(finalidxs)),output_file=cutterfile)
    return


def submit_cand_nsfrb(image,searched_image,TOAs,fname,uv_diag,dec_obs,args,suff,tsamp_use,DM_trials_use,cand_isot,cand_mjd,RA_axis,DEC_axis,RA_axis_2D,DEC_axis_2D,nsamps,injection_flag,postinjection_flag,lock=LOCK):
    """
    Modelled from dsa110-T3/dsaT3/T3_manager.submit_cand(); Given filename of trigger json,
    create DSACand and submit to scheduler for T3 processing
    """


    #(1) sort candidates
    canddict = dict()
    d_sort = client.submit(cc.sort_cands,fname,searched_image,TOAs,args.SNRthresh,RA_axis,DEC_xis,widthtrials,DM_trials_use,canddict,
                            np.abs(image.shape[1]-searched_image.shape[1]),
                            np.abs(image.shape[0]-searched_image.shape[0]),
                            cutterfile,0,args.maxcands,args.writeraw,lock=lock,priority=1,resources={'MEMORY': 10e9})

    #(2) clustering
    d_cluster = client.submit(cluster_manage,d_sort,image,nsamps,dec_obs,args,cutterfile,DM_trials_use,widthtrials,injection_flag,postinjection_flag,lock=lock,priority=1,resources={'MEMORY': 10e9})

    #(3) classifying
    d_classify = client.submit(classify_manage,d_cluster,image,nsamps,nchans,dec_obs,args,cutterfile,DM_trials_use,widthtrials,cand_isot,injection_flag,postinjection_flag,lock=lock,priority=1,resources={'MEMORY': 10e9})

    #(4) writing csvs and jsons for remaining cands (might not be any remaining cands)
    d_write = client.submit(writecand_manage,d_classify,image,args,DM_trials_use,widthtrials,suff,cand_isot,slow,imgdiff,injection_flag,postinjection_flag,lock=lock,priority=1,resources={'MEMORY': 10e9})

    #(5) sending alerts (slack, triggers, etc)
    d_trigger = client.submit(sendtrigger_manage,d_write,image,searched_image,args,uv_diag,dec_obs,slow,imgdiff,RA_axis,DEC_axis,DM_trials_use,widthtrials,output_dir,cand_isot,suff,cutterfile,injection_flag,postinjection_flag,lock=lock,priority=1,resources={'MEMORY': 10e9})
    
    #(6) archiving
    d_archive = client.submit(archive_manage,d_trigger,cand_isot,suff,cutterfile,injection_flag,postinjection_flag,lock=lock,priority=1,resources={'MEMORY': 10e9})

    if args.trigger:
        d_cs = client.submit(run_createstructure_nsfrb, d_trigger, key=f"run_createstructure_nsfrb-{d.trigname}", lock=lock, priority=1)  # create directory structure
        d_vc = client.submit(run_voltagecopy_nsfrb, d_cs, key=f"run_voltagecopy_nsfrb-{d.trigname}", lock=lock)  # copy voltages
        d_h5 = client.submit(run_hdf5copy_nsfrb, d_cs, key=f"run_hdf5copy_nsfrb-{d.trigname}", lock=lock)  # copy hdf5
        fut = client.submit(run_final_nsfrb, (d_h5, d_cs, d_vc), key=f"run_final_nsfrb-{d.trigname}", lock=lock)
        return fut
    return d_archive


def run_createstructure_nsfrb(d_future, lock=None):
    """ Use DSACand (after filplot) to decide on creating/copying files to candidate data area.
    """
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


def run_hdf5copy_nsfrb(d_future, lock=None):
    """ Given DSACand (after filplot), copy hdf5 files
    """
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

def run_voltagecopy_nsfrb(d_future, lock=None):
    """ Given DSACand (after filplot), copy voltage files.
    """
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


def run_final_nsfrb(ds, lock=None):
    """ Reduction task to handle all final tasks in graph.
    May also update etcd to notify of completion.
    """

#    d, d_po, d_hb, d_il, d_as = ds
    d, d_fm, d_vc = ds
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
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,pixperFWHM,nchans

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

from realtime.rtreader import rtread_cand
from nsfrb.config import NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,NSFRB_CANDDADA_SLOW_KEY,NSFRB_SRCHDADA_SLOW_KEY,NSFRB_TOADADA_SLOW_KEY,NSFRB_CANDDADA_IMGDIFF_KEY,NSFRB_SRCHDADA_IMGDIFF_KEY,NSFRB_TOADADA_IMGDIFF_KEY
def main(args):
    sys.stderr = open(error_file,"w")
    printlog("Starting T4 Manager (realtime candcutter)...",output_file=cutterfile)
    printlog("Adding ETCD watch on key "+ETCDKEY,output_file=cutterfile)
    ETCD.add_watch(ETCDKEY, etcd_to_queue)
    tasklist=[]
    counter = 0

    while True:

        printlog("Looking for cands in queue:" + str(QQUEUE),output_file=cutterfile)        
        fname = raw_cand_dir + str(QQUEUE.get())
        uv_diag = float(QQUEUE.get())#np.frombuffer(bytes.fromhex(QQUEUE.get()))[0]
        dec = float(QQUEUE.get())#np.frombuffer(bytes.fromhex(QQUEUE.get()))[0]
        img_shape = tuple(QQUEUE.get())
        img_search_shape = tuple(QQUEUE.get())
        printlog("Cand Cutter found cand file " + str(fname),output_file=cutterfile)
        
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
            tsamp_use = tsamp
            DM_trials_use = DM_trials
        cand_isot = fname[fname.index("candidates_")+11:fname.index(suff + ".csv")]

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

        cand_mjd = Time(cand_isot,format='isot').mjd
        injection_flag,postinjection_flag = cc.is_injection(cand_isot)
        RA_axis,DEC_axis,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs,pixperFWHM=pixperFWHM)
        RA_axis = RA_axis[-searched_image.shape[1]:]
        RA_axis_2D,DEC_axis_2D,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs,two_dim=True,pixperFWHM=pixperFWHM)
        RA_axis_2D = RA_axis_2D[:,-searched_image.shape[1]:]
        DEC_axis_2D = DEC_axis_2D[:,-searched_image.shape[1]:]
        nsamps = image.shape[2]

        #submit task
        submit_cand_nsfrb(image,searched_image,TOAs,fname,uv_diag,dec,args,suff,tsamp_use,DM_trials_use,cand_isot,cand_mjd,
                        RA_axis,DEC_axis,RA_axis_2D,DEC_axis_2D,nsamps,injection_flag,postinjection_flag)
        

        #client.submit(cc.candcutter_task,fname,uv_diag,dec,img_shape,img_search_shape,vars(args),resources={'MEMORY':10e9})
        if args.sleep > 0:
            printlog("Sleeping for " + str(args.sleep/60) + " minutes",output_file=cutterfile)
            time.sleep(args.sleep)

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
    parser.add_argument('--model_weights3D',type=str, help='Path to the model weights file for 3D classifying',default="/dataz/dsa110/nsfrb/dsa110-nsfrb-training/NN_train/enhanced3dcnn_weights_final.pth")
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
    args = parser.parse_args()
    
    main(args)

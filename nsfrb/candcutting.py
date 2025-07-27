import numpy as np
from nsfrb.config import nsamps as init_nsamps
from nsfrb.config import NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,NSFRB_CANDDADA_SLOW_KEY,NSFRB_SRCHDADA_SLOW_KEY,NSFRB_TOADADA_SLOW_KEY,NSFRB_CANDDADA_IMGDIFF_KEY,NSFRB_SRCHDADA_IMGDIFF_KEY,NSFRB_TOADADA_IMGDIFF_KEY
#from realtime.rtreader import rtread_cand
from nsfrb.planning import find_fast_vis_label
from nsfrb import pipeline
#from dsaT4 import T4_manager as T4m
from nsfrb.outputlogging import numpy_to_fits
import os
import socket
import time
import matplotlib
matplotlib.use('agg')
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
import argparse
from astropy.time import Time
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import glob
import csv
import copy

from nsfrb.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset
from nsfrb.classifying_with_time import classify_images_3D
from nsfrb.noise import init_noise,noise_update_all,get_noise_dict
import hdbscan
import copy
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from nsfrb.config import tsamp,tsamp_slow,fmin,fmax,nchans,nsamps,NUM_CHANNELS, CH0, CH_WIDTH, AVERAGING_FACTOR, IMAGE_SIZE, c, Lon,Lat, DM_tol,table_dir,tsamp_imgdiff
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack
from nsfrb.imaging import uv_to_pix
from nsfrb import plotting as pl
from simulations_and_classifications import generate_PSF_images as scPSF
from sklearn.metrics.pairwise import euclidean_distances

import os
import sys
from nsfrb.config import noise_dir,output_file,processfile,cutterfile,cuttertaskfile,error_file,inject_file,recover_file,binary_file, pixperFWHM,candplotfile,candplotfile_slow,candplotfile_imgdiff,candplotupdatefile
from nsfrb.config import freq_axis


#PSF-weighted distance measure

def PSF_dist_metric(p1,p2,PSFfunc):
    return PSFfunc(p2[0]-p1[0],p2[1]-p2[1])*euclidean_distances(p1,p2)

#initial spatial clustering based on psf shape
def psf_cluster(cands,PSF,output_file=cuttertaskfile,useTOA=False,perc=90):
    printlog(str(len(cands)) + " candidates",output_file=output_file)

    #make list for each param
    raidxs = []
    decidxs = []
    dmidxs = []
    widthidxs = []
    snridxs = []
    TOAflag = (len(cands[0]) == 6) and useTOA
    if TOAflag:
        printlog("Using TOA info for clustering",output_file=output_file)
        TOAs = []
    for i in range(len(cands)):
        raidxs.append(cands[i][0])
        decidxs.append(cands[i][1])
        dmidxs.append(cands[i][3])
        widthidxs.append(cands[i][2])
        if TOAflag:
            TOAs.append(cands[i][4])
        snridxs.append(cands[i][-1])
    raidxs = np.array(raidxs)
    decidxs = np.array(decidxs)
    dmidxs = np.array(dmidxs)
    widthidxs = np.array(widthidxs)
    snridxs = np.array(snridxs)
    if TOAflag:
        TOAs = np.array(TOAs)

    printlog("Done creating arrays of test data",output_file=output_file)
    if TOAflag:
        test_data=np.array([raidxs,decidxs,dmidxs,widthidxs,TOAs]).transpose()
    else:
        test_data=np.array([raidxs,decidxs,dmidxs,widthidxs]).transpose()

    #create psf binary map
    PSFbin = PSF>np.nanpercentile(PSF,perc)
    printlog("Done creating binary map",output_file=output_file)

    #for each candidate, see what other candidates included in psf 
    binned_dmidxs = []
    binned_widthidxs = []
    binned_snridxs = []
    binned_raidxs = []
    binned_decidxs = []
    if TOAflag:
        binned_TOAs = []
    for i in range(len(cands)):
        printlog(str(i) + " ; index: ",output_file=output_file)
        printlog(int(PSF.shape[0]//2)+np.array(decidxs-decidxs[i],int), output_file=output_file)
        printlog(int(PSF.shape[1]//2)+np.array(raidxs-raidxs[i],int),output_file=output_file)
        printlog(PSFbin[int(PSF.shape[0]//2)+np.array(decidxs-decidxs[i],int),int(PSF.shape[1]//2)+np.array(raidxs-raidxs[i],int)],output_file=output_file)
        inc_idxs = np.arange(len(cands))[PSFbin[int(PSF.shape[0]//2)+np.array(decidxs-decidxs[i],int),int(PSF.shape[1]//2)+np.array(raidxs-raidxs[i],int)]]
        #take the unique widths, dms, and TOAs and sum the snrs
        if TOAflag:
            unique_cands,unique_idxs = np.unique(np.array([dmidxs[inc_idxs],widthidxs[inc_idxs],TOAs[inc_idxs]]),axis=1,return_index=True)
        else:
            unique_cands,unique_idxs = np.unique(np.array([dmidxs[inc_idxs],widthidxs[inc_idxs]]),axis=1,return_index=True)
        binned_raidxs = np.concatenate([binned_raidxs,[raidxs[i]]*unique_cands.shape[1]])
        binned_decidxs = np.concatenate([binned_decidxs,[decidxs[i]]*unique_cands.shape[1]])
        binned_dmidxs = np.concatenate([binned_dmidxs,unique_cands[0,:]])
        binned_widthidxs = np.concatenate([binned_widthidxs,unique_cands[1,:]])
        if TOAflag:
            binned_TOAs = np.concatenate([binned_TOAs,unique_cands[2,:]])

        printlog("finished making binned indices",output_file=output_file)

        snrs_i = np.zeros(unique_cands.shape[1])
        for j in range(unique_cands.shape[1]):
            condition = np.logical_and(dmidxs[inc_idxs]==unique_cands[0,j],widthidxs[inc_idxs]==unique_cands[1,j])
            if TOAflag:
                condition = np.logical_and(condition,TOAs[inc_idxs]==unique_cands[2,j])
            snrs_i[j] += np.sum(snridxs[inc_idxs][condition])
        binned_snridxs = np.concatenate([binned_snridxs,snrs_i])
        printlog("finished getting binned snrs",output_file=output_file)
        """
        if TOAflag:
            condition = np.array([(dmidxs[j] in unique_cands[0,:]) and 
                              (widthidxs[j] in unique_cands[1,:]) and 
                              (TOAs[j] in unique_cands[2,:]) and
                              (list(binned_dmidxs).index(dmidxs[j])==
                                  list(binned_widthidxs).index(widthidxs[j])==
                                  np.argmin(np.abs(unique_cands[2,:]-TOAs[j]))) for j in inc_idxs])
        else:
            condition = np.array([(dmidxs[j] in unique_cands[0,:]) and
                              (widthidxs[j] in unique_cands[1,:]) and
                              (list(binned_dmidxs).index(dmidxs[j])==
                                  list(binned_widthidxs).index(widthidxs[j])) for j in inc_idxs])
        binned_snridxs= np.concatenate([binned_snridxs,snridxs[inc_idxs][condition]])
        """
    binned_raidxs = np.array(binned_raidxs)
    binned_decidxs = np.array(binned_decidxs)
    binned_dmidxs = np.array(binned_dmidxs)
    binned_widthidxs = np.array(binned_widthidxs)
    binned_snridxs = np.array(binned_snridxs)

    if TOAflag:
        binned_TOAs = np.array(binned_TOAs)
    
    if TOAflag:
        return binned_raidxs,binned_decidxs,binned_dmidxs,binned_widthidxs,binned_snridxs,binned_TOAs
    else:
        return binned_raidxs,binned_decidxs,binned_dmidxs,binned_widthidxs,binned_snridxs

#hdbscan clustering function; clusters in DM, width, RA, DEC space
def hdbscan_cluster(cands,min_cluster_size=50,dmt=[0]*16,wt=[0]*5,SNRthresh=1,plot=False,show=False,output_file=cuttertaskfile,PSF=None,min_samples=2,useTOA=False,perc=90,avgcluster=False,out_dir=""):
    printlog("WHY ISN'T IT STARTING?",output_file=output_file)
    #f = open(output_file,"a")
    printlog(str(len(cands)) + " candidates",output_file=output_file)
    TOAflag = (len(cands[0]) == 6) and useTOA
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True, min_samples=min_samples)

    #cluster in space first if PSF specified
    if PSF is not None:
        printlog("Clustering in space with PSF...",output_file=output_file)
        if TOAflag:
            raidxs,decidxs,dmidxs,widthidxs,snridxs,TOAs = psf_cluster(cands,PSF,output_file=output_file,useTOA=TOAflag,perc=perc)
        else:
            raidxs,decidxs,dmidxs,widthidxs,snridxs = psf_cluster(cands,PSF,output_file=output_file,useTOA=TOAflag,perc=perc)
        printlog((len(raidxs),len(decidxs),len(dmidxs),len(snridxs)),output_file)
        printlog(str(len(raidxs)) + " candidates remain after PSF clustering",output_file=output_file)
        """
        #cluster each unique position separately
        unique_cands,unique_idxs = np.unique(np.array([raidxs,decidxs]),axis=1,return_index=True)
        classes = np.zeros(len(raidxs),dtype=int)
        nclasses = 0
        printlog("Clustering in DM/Width/TOA...",output_file=output_file)
        for i in range(unique_cands.shape[1]):
            condition = np.logical_and(raidxs==unique_cands[0,i],decidxs==unique_cands[1,i])
            #printlog(str(i) + "," + str(np.sum(condition)),output_file=output_file)
            if sum(condition) >= 2:
                dmidxs_i = dmidxs[condition]
                widthidxs_i = widthidxs[condition]
                if TOAflag:
                    TOAs_i = TOAs[condition]
                
                if TOAflag:
                    test_data=np.array([dmidxs_i,widthidxs_i,TOAs_i]).transpose()
                else:
                    test_data=np.array([dmidxs_i,widthidxs_i]).transpose()

                #cluster data
                clusterer.fit(test_data)
                classes_i = clusterer.labels_
                classes_i[classes_i!=-1] += nclasses
                classes[condition] = classes_i
                nclasses += len(np.unique(classes_i))-(1 if -1 in classes_i else 0)
            else:
                classes[condition] = nclasses
                nclasses += 1
        """
    else:

        #make list for each param
        raidxs = []
        decidxs = []
        dmidxs = []
        widthidxs = []
        snridxs = []
        if TOAflag:
            printlog("Using TOA info for clustering",output_file=output_file)
            TOAs = []
        for i in range(len(cands)):
            raidxs.append(cands[i][0])
            decidxs.append(cands[i][1])
            dmidxs.append(cands[i][3])
            widthidxs.append(cands[i][2])
            if TOAflag:
                TOAs.append(cands[i][4])
            snridxs.append(cands[i][-1])
        raidxs = np.array(raidxs)
        decidxs = np.array(decidxs)
        dmidxs = np.array(dmidxs)
        widthidxs = np.array(widthidxs)
        snridxs = np.array(snridxs)
        if TOAflag:
            TOAs = np.array(TOAs)

    if TOAflag:
        test_data=np.array([raidxs,decidxs,dmidxs,widthidxs,TOAs]).transpose()
    else:
        test_data=np.array([raidxs,decidxs,dmidxs,widthidxs]).transpose()


    #cluster data
    clusterer.fit(test_data)
    classes = clusterer.labels_
        
    #print number of noise points
    noisepoints = np.sum(classes==-1)#clusterer.labels_==-1)
    printlog(str(noisepoints) + " noise points",output_file)

    nclasses = len(np.unique(classes))#clusterer.labels_))
    classnames = np.unique(classes)#clusterer.labels_)
    #classes = clusterer.labels_
    if -1 in classes:#clusterer.labels_:
        nclasses -= 1

    printlog(str(nclasses) + " unique classes",output_file)

    #get centroids
    #fcsv = open(cand_dir + "hdbscan_cluster_cands.csv","w")
    #csvwriter = csv.writer(fcsv)
    centroid_ras = []
    centroid_decs = []
    centroid_dms = []
    centroid_widths = []
    centroid_snrs = []
    if TOAflag:
        centroid_TOAs = []
    for k in classnames:
        if k != -1:
            if avgcluster:
                centroid_ras.append(np.average(raidxs[classes==k],weights=snridxs[classes==k]))
                centroid_decs.append(np.average(decidxs[classes==k],weights=snridxs[classes==k]))
                centroid_dms.append(np.average(dmidxs[classes==k],weights=snridxs[classes==k]))
                centroid_widths.append(np.average(widthidxs[classes==k],weights=snridxs[classes==k]))
                centroid_snrs.append(np.average(snridxs[classes==k],weights=snridxs[classes==k]))
                if TOAflag:
                    centroid_TOAs.append(np.average(TOAs[classes==k],weights=snridxs[classes==k]))
            
            else:
                centroid_ras.append(raidxs[classes==k][np.nanargmax(snridxs[classes==k])])
                centroid_decs.append(decidxs[classes==k][np.nanargmax(snridxs[classes==k])])
                centroid_dms.append(dmidxs[classes==k][np.nanargmax(snridxs[classes==k])])
                centroid_widths.append(widthidxs[classes==k][np.nanargmax(snridxs[classes==k])])
                centroid_snrs.append(snridxs[classes==k][np.nanargmax(snridxs[classes==k])])
                if TOAflag:
                    centroid_TOAs.append(TOAs[classes==k][np.nanargmax(snridxs[classes==k])])
            #csvwriter.writerow([centroid_ras[-1],centroid_decs[-1],centroid_widths[-1],centroid_dms[-1],centroid_snrs[-1]])
    #fcsv.close()
    centroid_ras = np.array(centroid_ras)
    centroid_decs = np.array(centroid_decs)
    centroid_dms = np.array(centroid_dms)
    centroid_widths = np.array(centroid_widths)
    centroid_snrs = np.array(centroid_snrs)
    if TOAflag:
        centroid_TOAs = np.array(centroid_TOAs)
    printlog("Done gathering centroids",output_file)

    if TOAflag:
        centroid_cands = [(centroid_ras[i],centroid_decs[i],centroid_widths[i],centroid_dms[i],centroid_TOAs[i],centroid_snrs[i]) for i in range(len(centroid_ras))]
    else:
        centroid_cands = [(centroid_ras[i],centroid_decs[i],centroid_widths[i],centroid_dms[i],centroid_snrs[i]) for i in range(len(centroid_ras))]
    printlog("Done gathering centroid cands",output_file)

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


        plt.savefig(out_dir + "hdbscan_cluster_plot.png")
        if show:
            plt.show()
        else:
            plt.close()
    #f.close()
    printlog("finished clustering",output_file)
    if TOAflag:
        return classes,centroid_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs,centroid_TOAs
    else:
        return classes,centroid_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs



#code to cutout subimages
freq_axis = np.linspace(fmin,fmax,nchans)
def get_subimage(image_tesseract,ra_idx,dec_idx,subimgpix=11,save=False,prefix="candidate_stamp",plot=False,output_file=cutterfile,output_dir="",dm=None,dmidx=None,tsamp=tsamp,freq_axis=freq_axis):
    """
    if output_file != "":
        fout = open(output_file,"a")
    else:
        fout = sys.stdout
    """
    gridsize = image_tesseract.shape[0]
    fname = output_dir + prefix + "_" + str(ra_idx) + "_" + str(dec_idx)
    if subimgpix%2 == 0:
        printlog("subimgpix must be odd",output_file=output_file)
        #if output_file != "":
        #    fout.close()
        return None


    #dedisperse if given a dm
    if dm is not None:
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
                                                   mode='edge')#'constant',
                                                   #constant_values=np.nan)

    #cut out subimage
    minraidx = int(gridsize + ra_idx - subimgpix//2)#np.max([ra_idx - subimgpix//2,0])
    maxraidx = int(gridsize + ra_idx + subimgpix//2 + 1)#np.min([ra_idx + subimgpix//2 + 1,gridsize-1])
    mindecidx = int(gridsize + dec_idx - subimgpix//2)#np.max([dec_idx - subimgpix//2,0])
    maxdecidx = int(gridsize + dec_idx + subimgpix//2 + 1)#np.min([dec_idx + subimgpix//2 + 1,gridsize-1])

    #print(minraidx_cut,maxraidx_cut,mindecidx_cut,maxdecidx_cut)
    printlog(str((minraidx,maxraidx,mindecidx,maxdecidx)),output_file=output_file)

    image_cutout = image_tesseract_dm[mindecidx:maxdecidx,minraidx:maxraidx,:,:]

    if save:
        np.save(fname,image_cutout)

    if plot:
        plt.figure(figsize=(12,12))
        plt.imshow(image_cutout.mean((2,3)),aspect='auto')
        plt.show()
    #if output_file != "":
    #fout.close()
    return image_cutout




#checks injection file to see if a candidate is an injection
def is_injection(isot,inject_file=inject_file,tsamp=tsamp,nsamps=nsamps,realtime=False):
    #check if the candidate is an injection
    injection = False
    postinjection = False
    with open(inject_file,"r") as csvfile:
        re = csv.reader(csvfile,delimiter=',')
        i = 0
        for row in re:
            if i != 0:
                if realtime:
                    if row[0] == isot or ((Time(row[0][:-1],format='isot').mjd - Time(isot,format='isot').mjd)*86400 <= (tsamp*nsamps/1000)):
                        injection = True
                        break
                    elif (tsamp<134*5) and ((Time(row[0][:-1],format='isot').mjd - Time(isot,format='isot').mjd)*86400 <= 2*(tsamp*nsamps/1000)):
                        postinjection = False
                        break
                else:
                    if row[0] == isot or row[0][:-1] == Time(Time(isot,format='isot').mjd - (tsamp*nsamps/1000/86400),format='mjd').isot[:-1]:
                        injection = True
                        if row[0][:-1] == Time(Time(isot,format='isot').mjd - (tsamp*nsamps/1000/86400),format='mjd').isot[:-1]:
                            postinjection = True
                        break
            i += 1
    csvfile.close()
    return injection,postinjection


def read_candfile(fname):
    finalcands = []
    raw_cand_names = []
    with open(fname,"r") as csvfile:
        re = csv.reader(csvfile,delimiter=',')
        for r in re:
            if 'candname' not in r:
                finalcands.append(np.array(r[1:],dtype=float))
                raw_cand_names.append(r[0])
    csvfile.close()
    return raw_cand_names,finalcands

def sort_cands(fname,image_tesseract_binned,TOAs,SNRthresh,RA_axis,DEC_axis,widthtrials,DM_trials,canddict,raidx_offset=0,decidx_offset=0,output_file=cutterfile,dm_offset=0,maxcands=np.inf,writeraw=False,dm0_only=False,dm0_ex=False,w1_only=False,searchradius=np.inf):

    t1 = time.time()
    printlog("Searching for candidates with S/N > " + str(SNRthresh) + "...",output_file=output_file)
    #find candidates above SNR threshold
    #condition = (image_tesseract_binned>=SNRthresh).flatten()
    #ncands = np.sum(condition)
    #canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs=np.unravel_index(np.arange(gridsize_DEC*gridsize_RA*ndms*nwidths)[condition],(gridsize_DEC,gridsize_RA,nwidths,ndms))#[1].shape


    canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs = np.nonzero(image_tesseract_binned>=SNRthresh)
    if ~np.isinf(maxcands) and len(canddec_idxs)>maxcands:
        printlog("Limiting to max " + str(maxcands) + " candidates",output_file=output_file)
        candsnrs = image_tesseract_binned[canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs]
        idxs = np.argsort(candsnrs)[-maxcands:]
        

        canddec_idxs = canddec_idxs[idxs]
        candra_idxs = candra_idxs[idxs]
        candwid_idxs = candwid_idxs[idxs]
        canddm_idxs = canddm_idxs[idxs]
    if dm0_only:
        printlog("Keeping only DM=0 candidates",output_file=output_file)
        idxs = np.arange(len(canddm_idxs))[canddm_idxs==0]
        canddec_idxs = canddec_idxs[idxs]
        candra_idxs = candra_idxs[idxs]
        candwid_idxs = candwid_idxs[idxs]
        canddm_idxs = canddm_idxs[idxs]
    elif dm0_ex:
        printlog("Removing DM=0 candidates",output_file=output_file)
        idxs = np.arange(len(canddm_idxs))[canddm_idxs>0]
        canddec_idxs = canddec_idxs[idxs]
        candra_idxs = candra_idxs[idxs]
        candwid_idxs = candwid_idxs[idxs]
        canddm_idxs = canddm_idxs[idxs]

    if w1_only:
        printlog("Keeping only width=1 sample candidates",output_file=output_file)
        idxs = np.arange(len(candwid_idxs))[candwid_idxs==0]
        canddec_idxs = canddec_idxs[idxs]
        candra_idxs = candra_idxs[idxs]
        candwid_idxs = candwid_idxs[idxs]
        canddm_idxs = canddm_idxs[idxs]

    if ~np.isinf(searchradius):
        printlog("limiting to candidates w/in " + str(searchradius) + "degrees of center",output_file=output_file)
        ra_cntr = RA_axis[int(len(RA_axis)//2)]
        dec_cntr = DEC_axis[int(len(DEC_axis)//2)]
        idxs = np.arange(len(candra_idxs))[np.logical_and(np.abs(RA_axis[candra_idxs.astype(int)]-ra_cntr)<searchradius,np.abs(DEC_axis[canddec_idxs.astype(int)]-dec_cntr)<searchradius)]
        canddec_idxs = canddec_idxs[idxs]
        candra_idxs = candra_idxs[idxs]
        candwid_idxs = candwid_idxs[idxs]
        canddm_idxs = canddm_idxs[idxs]


    printlog(fname + " CANDS::::",output_file=output_file)
    printlog(canddec_idxs,output_file=output_file)
    printlog(candra_idxs,output_file=output_file)
    printlog(candwid_idxs,output_file=output_file)
    printlog(canddm_idxs,output_file=output_file)
    #fout.close()
    ncands = len(canddec_idxs)
    #print(len(DEC_axis),np.max(canddec_idxs),len(RA_axis),np.max(candra_idxs),file=fout)
    #fout.close()
    canddecs = DEC_axis[canddec_idxs]
    candras = RA_axis[candra_idxs]
    candwids = widthtrials[candwid_idxs]
    canddms = DM_trials[canddm_idxs]
    candsnrs = image_tesseract_binned[canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs]#.flatten()[condition]
    candTOAs = TOAs[canddec_idxs,candra_idxs,candwid_idxs,canddm_idxs]
    candidxs = [(raidx_offset + candra_idxs[i],decidx_offset + canddec_idxs[i],candwid_idxs[i],dm_offset + canddm_idxs[i],candTOAs[i],candsnrs[i]) for i in range(ncands)]
    cands = [(candras[i],canddecs[i],candwids[i],canddms[i],candTOAs[i],candsnrs[i]) for i in range(ncands)]

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
    canddict['TOAs'] = copy.deepcopy(candTOAs)
    printlog("Time for sorting candidates: " + str(time.time()-t1) + " s",output_file=output_file)
    


    
    #write raw candidates to csv
    if writeraw:
        printlog("Writing to file " + fname,output_file=output_file)
        csvfile = open(fname,"w")
        wr = csv.writer(csvfile,delimiter=',')
        wr.writerow(["candname","RA index","DEC index","WIDTH index", "DM index", "TOA", "SNR"])
        finalcands = []
        for i in range(len(candidxs)):
            wr.writerow(np.concatenate([[i],np.array(candidxs[i][:-1],dtype=int),[candidxs[i][-1]]]))
            finalcands.append(np.concatenate([np.array(candidxs[i][:-1],dtype=int),[candidxs[i][-1]]]))
        csvfile.close()
        printlog("Done",output_file=output_file)
    else:
        finalcands = []
        for i in range(len(candidxs)):
            finalcands.append(np.concatenate([np.array(candidxs[i][:-1],dtype=int),[candidxs[i][-1]]]))

    return np.arange(len(candidxs)).astype(str),finalcands

#classifier format
from torchvision import transforms
from PIL import Image
def img_to_classifier_format(img,candname,output_dir):
    #if not square, pad with zeros
    gridsize_DEC,gridsize_RA,nchans = img.shape
    if gridsize_DEC > gridsize_RA:
        img = np.pad(img,((0,0),(int((gridsize_DEC-gridsize_RA)//2),(gridsize_DEC-gridsize_RA) - int((gridsize_DEC-gridsize_RA)//2)),(0,0))) 
    gridsize_DEC,gridsize_RA,nchans = img.shape
    img_class_format = np.zeros_like(img,dtype=np.float64)

    for i in range(nchans):
        avg_freq = CH0 + CH_WIDTH * i * AVERAGING_FACTOR
        filename = f'{candname}_{avg_freq:.2f}_MHz.png'
        plt.imsave(output_dir + filename,img[:,:,i],cmap='gray')
        newimg = Image.open(output_dir + filename).convert('L')
        os.system("rm " + output_dir + filename)

        img_class_format[:,:,i] = transforms.ToTensor()(newimg)[0] 
    return img_class_format








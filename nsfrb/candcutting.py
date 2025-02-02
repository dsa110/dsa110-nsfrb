import numpy as np
from nsfrb.planning import find_fast_vis_label
from nsfrb import pipeline
from dsaT4 import T4_manager as T4m
from nsfrb.outputlogging import numpy_to_fits
import os
import jax
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
#from gen_dmtrials_copy import gen_dm
import argparse
from astropy.time import Time
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import glob
import csv
import copy

from nsfrb.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset
from nsfrb.noise import init_noise,noise_update_all,get_noise_dict
import hdbscan
import copy
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from nsfrb.config import tsamp,fmin,fmax,nchans,nsamps,NUM_CHANNELS, CH0, CH_WIDTH, AVERAGING_FACTOR, IMAGE_SIZE, c, Lon,Lat, DM_tol
from nsfrb.searching import gen_dm_shifts,widthtrials,DM_trials,gen_boxcar_filter,default_PSF
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack
from nsfrb.imaging import uv_to_pix
from nsfrb import plotting as pl
from simulations_and_classifications import generate_PSF_images as scPSF
from sklearn.metrics.pairwise import euclidean_distances

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
import os
import sys
"""
cwd = os.environ['NSFRBDIR']
cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/" #cwd + "-candidates/"
vis_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-fast-visibilities/"
cutterfile = cwd + "-logfiles/candcutter_log.txt"
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = cwd + "/src/.pipestatus.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = cwd + "/scripts/script_flags/searchlog_flags.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_flags/searchlog_flags.txt"
output_file = cwd + "-logfiles/run_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
processfile = cwd + "-logfiles/process_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt"
cutterfile = cwd + "-logfiles/candcutter_log.txt"
cuttertaskfile = cwd + "-logfiles/candcuttertask_log.txt"
flagfile = cwd + "/process_server/process_flags.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_flags.txt"
raw_cand_dir = cand_dir + "raw_cands/"#cwd + "-candidates/raw_cands/"
backup_cand_dir = cand_dir + "backup_raw_cands/"#cwd + "-candidates/backup_raw_cands/"
final_cand_dir = cand_dir + "final_cands/"#cwd + "-candidates/final_cands/"
inject_dir = inject_file = cwd + "-injections/"
error_file = cwd + "-logfiles/error_log.txt"
inject_file = cwd + "-injections/injections.csv"
recover_file = cwd + "-injections/recoveries.csv"
training_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-training/"
psf_dir = cwd + "-PSF/"
img_dir = cwd + "-images/"
sys.path.append(cwd + "/") #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
"""
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file


freq_axis = np.linspace(fmin,fmax,nchans)
corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append,wraps_append,wraps_no_append = gen_dm_shifts(DM_trials,freq_axis,tsamp,nsamps,outputwraps=True)
full_boxcar_filter = gen_boxcar_filter(widthtrials,nsamps)
tDM_max = (4.15)*np.max(DM_trials)*((1/np.min(freq_axis)/1e-3)**2 - (1/np.max(freq_axis)/1e-3)**2) #ms
maxshift = int(np.ceil(tDM_max/tsamp))


#PSF-weighted distance measure
"""
def PSF_dist_metric(*test_points,PSF=default_PSF.mean((2,3))):
    cntr_x,cntr_y = PSF.shape[0]//2,PSF.shape[1]//2
    offset_x,offset_y = test_points[0][1] - test_points[1][1], test_points[0][0] - test_points[1][0]
    weight = PSF[int(cntr_x + offset_x),int(cntr_y + offset_y)]
    return weight*np.sqrt(np.sum((test_points[1]-test_points[0])**2))
"""

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
def hdbscan_cluster(cands,min_cluster_size=50,dmt=[0]*16,wt=[0]*5,SNRthresh=1,plot=False,show=False,output_file=cuttertaskfile,PSF=None,min_samples=2,useTOA=False,perc=90):
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
            centroid_ras.append(np.average(raidxs[classes==k],weights=snridxs[classes==k]))
            centroid_decs.append(np.average(decidxs[classes==k],weights=snridxs[classes==k]))
            centroid_dms.append(np.average(dmidxs[classes==k],weights=snridxs[classes==k]))
            centroid_widths.append(np.average(widthidxs[classes==k],weights=snridxs[classes==k]))
            centroid_snrs.append(np.average(snridxs[classes==k],weights=snridxs[classes==k]))
            if TOAflag:
                centroid_TOAs.append(np.average(TOAs[classes==k],weights=snridxs[classes==k]))
            """
            centroid_ras.append((np.nansum((snridxs*raidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_decs.append((np.nansum((snridxs*decidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_dms.append((np.nansum((snridxs*dmidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_widths.append((np.nansum((snridxs*widthidxs)[classes==k])/np.nansum(snridxs[classes==k])))
            centroid_snrs.append(np.nansum((snridxs*snridxs)[classes==k])/np.nansum(snridxs[classes==k]))
            """
            
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
        centroid_cands = [(centroid_ras[i],centroid_decs[i],centroid_widths[i],centroid_dms[i],centroid_snrs[i],centroid_TOAs[i]) for i in range(len(centroid_ras))]
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


        plt.savefig(cand_dir + "hdbscan_cluster_plot.png")
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
def get_subimage(image_tesseract,ra_idx,dec_idx,subimgpix=11,save=False,prefix="candidate_stamp",plot=False,output_file=cutterfile,output_dir=cand_dir,corr_shifts=corr_shifts_all_no_append,tdelays_frac=tdelays_frac_no_append,dm=None,dmidx=None,tsamp=tsamp,freq_axis=freq_axis):
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
    if dmidx is not None: #(corr_shift is not None) and (tdelay_frac is not None) and (dmidx is not None): 
        image_tesseract_dm = quick_dedisp(image_tesseract,corr_shifts,tdelays_frac,DM_idx=dmidx)
    elif dm is not None:
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


#this is a quick implementation of dedispersion meant only for cutouts, classification, and injection
def quick_dedisp(sourceimg,corr_shifts=corr_shifts_all_no_append,tdelays_frac=tdelays_frac_no_append,wraps=wraps_no_append,DM_idx=0):
    #return (((np.take_along_axis(image_pixel.repeat(2,axis=3),indices=corr_shifts,axis=2))*tdelay_frac).reshape((image_pixel.shape[0],image_pixel.shape[1],image_pixel.shape[2],image_pixel.shape[3],2))).mean(4)
    print("quick dedisp start:",sourceimg.shape)
    
    gridsize_DEC,gridsize_RA = sourceimg.shape[:2]
    print("gsizes:",gridsize_DEC,gridsize_RA )
    sourceimg_dm = ((np.take_along_axis(sourceimg[:,:,:,np.newaxis,:].repeat(1,axis=3).repeat(2,axis=4),indices=corr_shifts[:gridsize_DEC,:gridsize_RA,:,DM_idx:DM_idx+1,:],axis=2))*tdelays_frac[:gridsize_DEC,:gridsize_RA,:,DM_idx:DM_idx+1,:])[:,:,:,0,:]
    print("dedipsed",sourceimg_dm.shape)
    #zero out anywhere that was wrapped
    sourceimg_dm[wraps[:,:,:,DM_idx,:].repeat(sourceimg.shape[0],axis=0).repeat(sourceimg.shape[1],axis=1)] = 0
    print("zeroed")
    #now average the low and high shifts 
    sourceimg_dm = (sourceimg_dm.reshape(tuple(list(sourceimg.shape) + [2])).sum(4))[:,:,::-1,:]
    return sourceimg_dm



#this is a copy of the jax binning function which runs on a single pixel on the CPU. it does not normalize by off-pulse noise and is
#only meant for classification purposes
def quick_snr_fft(image_pixel,width):
    boxcar = np.zeros((1,1,image_pixel.shape[0],1))
    boxcar[:,:,len(boxcar)//2 -width//2 - 2:len(boxcar)//2 -width//2 +width- 2,:] = 1
    return np.nan_to_num(np.real(np.fft.ifftshift(
                                            np.fft.ifft(
                                                np.fft.fft(image_pixel,
                                                            n=image_pixel.shape[2],
                                                            axis=2,norm='backward')*np.fft.fft(boxcar,
                                                            n=image_pixel.shape[2],axis=2,norm='backward'),
                                                        n=image_pixel.shape[2],
                                                        axis=2,norm='backward'),axes=2)),
                                            nan=0,posinf=0,neginf=0)



#checks injection file to see if a candidate is an injection
def is_injection(isot,inject_file=inject_file):
    #check if the candidate is an injection
    injection = False
    with open(inject_file,"r") as csvfile:
        re = csv.reader(csvfile,delimiter=',')
        i = 0
        for row in re:
            if i != 0:
                if row[0] == isot:
                    injection = True
                    break
            i += 1
    csvfile.close()
    return injection


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



#classifier format
from torchvision import transforms
from PIL import Image
def img_to_classifier_format(img,candname,output_dir):
    img_class_format = np.zeros_like(img,dtype=np.float64)
    gridsize_DEC,gridsize_RA,nchans = img.shape
    for i in range(nchans):
        avg_freq = CH0 + CH_WIDTH * i * AVERAGING_FACTOR
        filename = f'{candname}_{avg_freq:.2f}_MHz.png'
        plt.imsave(output_dir + filename,img[:,:,i],cmap='gray')
        newimg = Image.open(output_dir + filename).convert('L')
        os.system("rm " + output_dir + filename)

        img_class_format[:,:,i] = transforms.ToTensor()(newimg)[0] 
    return img_class_format



#main cand cutter task function
def candcutter_task(fname,uv_diag,dec_obs,args):
    """
    Main task to obtain cutouts
    """
    #for each candidate get the isot and find the corresponding image
    cand_isot = fname[fname.index("candidates_")+11:fname.index(".csv")]
    cand_mjd = Time(cand_isot,format='isot').mjd
    injection_flag = is_injection(cand_isot)
    
    #read cand file
    raw_cand_names,finalcands = read_candfile(fname)
    #raw_cand_names = raw_cand_names[:100]
    #finalcands = finalcands[:100]

    #prune candidates with infinite signal-to-noise for clustering
    cands_noninf = []
    for fcand in finalcands:
        if not np.isinf(fcand[-1]): cands_noninf.append(fcand)

    #take out low S/N percentile if specified
    if args['percentile'] > 0:
        candsnrs = np.array([fcand[-1] for fcand in cands_noninf])
        snrp = np.nanpercentile(candsnrs,args['percentile'])

        cands_perc = []
        for fcand in cands_noninf:
            if fcand[-1] > snrp: cands_perc.append(fcand)
            #if len(cands_perc) > 10: break            
            
        cands_noninf = cands_perc
        printlog(str(len(cands_noninf)) + " candidates remaining after " + str(args['percentile']) + "th percentile cutoff",output_file=cutterfile)

    #cut by S/N if still too many
    if len(cands_noninf) >args['maxcands']:
        printlog(cand_isot + "has too many candidates to process (" + str(len(cands_noninf)) + ">" + str(args['maxcands']) + ") limit...",output_file=cutterfile)
        sortedcands = list(np.array(cands_noninf)[np.argsort(np.array(cands_noninf)[:,-1])[::-1],:])
        cands_noninf = sortedcands[:int(args['maxcands'])]
        printlog("done, cut to " + str(len(cands_noninf)) + " candidates",output_file=cutterfile)
    """
    #confirm number of cands less than max
    if len(finalcands) >args['maxcands']: 
        printlog(cand_isot + "has too many candidates to process (" + str(len(finalcands)) + ">" + str(args['maxcands']) + "), please adjust S/N threshold",output_file=cutterfile)
        return
    """
        


    """
    finalcands = []
    raw_cand_names = []
    with open(fname,"r") as csvfile:
        re = csv.reader(csvfile,delimiter=',')
        for r in re:
            if 'candname' not in r:
                finalcands.append(np.array(r[1:],dtype=float))
                raw_cand_names.append(r[0])
    csvfile.close()
    """
    finalcands = copy.deepcopy(cands_noninf)
    finalidxs = np.arange(len(finalcands),dtype=int)

    #if getting cutouts, read image
    try:
        image = np.load(raw_cand_dir + cand_isot + ".npy")
        searched_image = np.load(raw_cand_dir + cand_isot + "_searched.npy")
    except Exception as e:
        printlog("No image found for candidate " + cand_isot,output_file=cutterfile)
        return
    RA_axis,DEC_axis,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs)
    RA_axis = RA_axis[-image.shape[1]:]
    RA_axis_2D,DEC_axis_2D,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs,two_dim=True)
    RA_axis_2D = RA_axis_2D[:,-image.shape[1]:]
    DEC_axis_2D = DEC_axis_2D[:,-image.shape[1]:]

    #RA_axis = RA_axis[int((len(RA_axis)-image.shape[1])//2):int((len(RA_axis)-image.shape[1])//2) + image.shape[1]]
    #PSF = scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),image.shape[0]//2,True,nsamps)

    #get DM trials from file
    """
    DMtrials = np.load(cand_dir + "DMtrials.npy")
    widthtrials = np.load(cand_dir + "widthtrials.npy")
    SNRthresh = float(np.load(cand_dir +"SNRthresh.npy"))
    corr_shifts = np.load(cand_dir+"DMcorr_shifts.npy")
    tdelays_frac = np.load(cand_dir+"DMdelays_frac.npy")
    """

    #start clustering
    useTOA=args['useTOA'] and len(finalcands[0])==6
    if args['cluster'] and len(finalidxs)>=args['mincluster']:
        printlog("clustering with HDBSCAN...",output_file=cutterfile)
        """
        #prune candidates with infinite signal-to-noise for clustering
        cands_noninf = []
        for fcand in finalcands:
            if not np.isinf(fcand[-1]): cands_noninf.append(fcand)

        #take out low S/N percentile if specified
        if args['percentile'] > 0:
            candsnrs = np.array([fcand[-1] for fcand in finalcands])
            snrp = np.nanpercentile(candsnrs,args['percentile'])

            cands_perc = []
            for fcand in cands_noninf:
                if fcand[-1] > snrp: cands_perc.append(fcand)
                #if len(cands_perc) > 10: break
            cands_noninf = cands_perc
            printlog(str(len(cands_noninf)) + " candidates remaining after " + str(args['percentile']) + "th percentile cutoff",output_file=cutterfile)
        """
        #clustering with hdbscan
        if args['psfcluster']:
            PSF,PSF_params = scPSF.manage_PSF(scPSF.make_PSF_dict(),(2*image.shape[0])+1,dec_obs,nsamps=nsamps)#scPSF.generate_PSF_images(psf_dir,np.nanmean(DEC_axis),image.shape[0],True,nsamps).mean((2,3))
            #PSF = PSF[:,int((PSF.shape[1]-image.shape[1])//2):int((PSF.shape[1]-image.shape[1])//2) + image.shape[1],:,:]
            PSF = PSF.mean((2,3))
            printlog("PSF shape for clustering:" + str(PSF.shape),output_file=cutterfile)
        else:
            PSF = None
        if useTOA:
            classes,cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs,centroid_TOAs = hdbscan_cluster(cands_noninf,min_cluster_size=args['mincluster'],min_samples=args['minsamples'],dmt=DM_trials,wt=widthtrials,plot=False,show=False,SNRthresh=args['SNRthresh'],PSF=PSF,useTOA=True,perc=args['psfpercentile'])
        else:
            classes,cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs = hdbscan_cluster(cands_noninf,min_cluster_size=args['mincluster'],min_samples=args['minsamples'],dmt=DM_trials,wt=widthtrials,plot=False,show=False,SNRthresh=args['SNRthresh'],PSF=PSF,perc=args['psfpercentile'])
        printlog("done, made " + str(len(cluster_cands)) + " clusters",output_file=cutterfile)
        printlog(classes,output_file=cutterfile)
        printlog(cluster_cands,output_file=cutterfile)

        finalidxs = np.arange(len(cluster_cands),dtype=int)
        finalcands = cluster_cands
        
    #cut by S/N if still too many
    if len(finalcands) >args['maxcands_postcluster']:
        printlog(cand_isot + "has too many candidates to process post-clustering (" + str(len(finalcands)) + ">" + str(args['maxcands_postcluster']) + ") limit...",output_file=cutterfile)
        sortedcands = list(np.array(finalcands)[np.argsort(np.array(finalcands)[:,-1])[::-1],:])
        finalcands = sortedcands[:int(args['maxcands_postcluster'])]
        finalidxs = np.arange(len(finalcands),dtype=int)
        printlog("done, cut to " + str(len(finalcands)) + " candidates",output_file=cutterfile)
    


    if args['classify']:
        if args['subimgpix'] == image.shape[0]:
            printlog("Using full image for classification and cutouts",output_file=cutterfile)
            data_array = (img_to_classifier_format(image.mean(2),cand_isot,img_dir)[np.newaxis,:,:,:]).repeat(len(finalcands),axis=0)
        else:
            #make a binned copy for each candidate
            data_array = np.zeros((len(finalcands),args['subimgpix'],args['subimgpix'],image.shape[3]),dtype=np.float64)
            for j in range(len(finalcands)):
                printlog(finalcands[j],output_file=cutterfile)
                #subimg = quick_snr_fft(get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args['subimgpix'],corr_shift=corr_shifts[:,:,:,int(finalcands[j][3]):int(finalcands[j][3])+1,:],tdelay_frac=tdelays_frac[:,:,:,int(finalcands[j][3]):int(finalcands[j][3])+1,:]),widthtrials[int(finalcands[j][2])])
                #subimg = quick_snr_fft(get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args['subimgpix'],dmidx=int(finalcands[j][3])),widthtrials[int(finalcands[j][2])])

                #don't need to dedisperse(?)
                subimg = get_subimage(image,int(finalcands[j][0]),int(finalcands[j][1]),save=False,subimgpix=args['subimgpix'])
                if useTOA:
                    printlog("using TOA...",output_file=cutterfile)
                    loc = int(finalcands[j][4])
                    printlog("got loc...",output_file=cutterfile)
                    wid = widthtrials[int(finalcands[j][2])]
                    printlog("got wid...",output_file=cutterfile)
                    data_array[j,:,:,:] = img_to_classifier_format(subimg[:,:,int(loc+1-(wid//2)):int(loc+1-(wid//2) + wid),:].mean(2),cand_isot+"_"+str(j),img_dir)
                    printlog("img to classifier formatd done...",output_file=cutterfile)
                else:
                    data_array[j,:,:,:] = img_to_classifier_format(subimg.mean(2),cand_isot+"_"+str(j),img_dir)  #.mean(2)#subimg[:,:,np.argmax(subimg.sum((0,1,3))),:]
                printlog("cand shape:" + str(data_array[j,:,:,:].shape),output_file=cutterfile)
            
        #reformat for classifier
        #transposed_array = np.transpose(data_array, (0,3,1,2))#cands x frequencies x RA x DEC
        #new_shape = (data_array.shape[0], data_array.shape[3], data_array.shape[1], data_array.shape[2])
        merged_array = np.transpose(data_array, (0,3,1,2)) #transposed_array.reshape(new_shape)

        printlog("shape input to classifier:" + str(merged_array.shape),output_file=cutterfile)
        #run classifier
        predictions, probabilities = classify_images(merged_array, args['model_weights'], verbose=args['verbose'])
        printlog(predictions,output_file=cutterfile)
        printlog(probabilities,output_file=cutterfile)

        #only save bursts likely to be real
        #finalidxs = finalidxs[~np.array(predictions,dtype=bool)]

    #if its an injection write the highest SNR candidate to the injection tracker
    if injection_flag:
        with open(recover_file,"a") as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            for j in finalidxs:
                wr.writerow([cand_isot,DM_trials[int(finalcands[j][3])],widthtrials[int(finalcands[j][2])],finalcands[j][-1],(None if not args['classify'] else predictions[j]),(None if not args['classify'] else probabilities[j])])
        csvfile.close()


    #make final directory for candidates
    os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot)

    #write final candidates to csv
    prefix = "NSFRB"
    lastname = None     #once we have etcd, change to 'names.get_lastname()'
    allcandnames = []
    csvfile = open(final_cand_dir+ str("injections" if injection_flag else "candidates")  + "/" + cand_isot + "/final_candidates_" + cand_isot + ".csv","w")
    wr = csv.writer(csvfile,delimiter=',')
    hdr = ["candname","RA index","DEC index","WIDTH index", "DM index", "SNR"]
    if useTOA: hdr += ["TOA"]
    if args['classify']: hdr += ["PROB"]
    wr.writerow(hdr)
    sysstdout = sys.stdout
    for j in finalidxs:#range(len(finalidxs)):
        with open(cutterfile,"a") as sys.stdout:
            lastname = names.increment_name(cand_mjd,lastname=names.get_lastname())
        sys.stdout = sysstdout
        if args['classify']:
            wr.writerow(np.concatenate([[lastname],np.array(finalcands[j][:-1],dtype=int),[finalcands[j][-1]],[probabilities[j]]]))
        else:
            wr.writerow(np.concatenate([[lastname],np.array(finalcands[j][:-1],dtype=int),[finalcands[j][-1]]]))
        allcandnames.append(prefix + lastname)
    csvfile.close()


    #make subdirectories for candidates
    for j in finalidxs:

        lastname = allcandnames[j]
        #make folder for each candidate
        os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + lastname)
        os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + lastname + "/voltages")

    #get image cutouts and write to file
    if args['cutout'] or args['train']:
        for j in finalidxs:#range(len(finalidxs)):
            if args['subimgpix'] != image.shape[0]:
                subimg = get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args['subimgpix'])#[:,:,int(finalcands[j][2]),:]
            else:
                subimg = image
            lastname = allcandnames[j]

            if args['train']:
                printlog(training_dir+ str("simulated/" if injection_flag else "data/") + cand_isot + "_" + str(j),output_file=cutterfile)
                for k in range(subimg.shape[3]):
                    filepath = training_dir+ str("simulated/" if injection_flag else "data/") + cand_isot + "_" + str(j) + "_subband_avg_{F:.2f}_MHz".format(F=CH0 + CH_WIDTH * k * AVERAGING_FACTOR) + ".png"
                    printlog(filepath,output_file=cutterfile)
                    if useTOA:
                        loc = int(finalcands[j][4])
                        wid = widthtrials[int(finalcands[j][2])]
                        plt.imsave(filepath, subimg[:,:,int(loc+1-(wid//2)):int(loc+1-(wid//2) + wid),k].mean(2), cmap='gray')
                    else:
                        plt.imsave(filepath, subimg[:,:,:,k].mean(2), cmap='gray')
                np.save(training_dir+ str("simulated/" if injection_flag else "data/") + cand_isot + "_" + str(j) + ".npy",subimg)

                f = open(training_dir+ str("simulated/" if injection_flag else "data/") + "labels.txt","a")
                f.write(cand_isot + "_" + str(j) + ",-1,end\n")
                f.close()


            #make folder for each candidate
            os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + lastname)
            os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + lastname + "/voltages")
            np.save(final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + lastname + "/" + lastname + ".npy",subimg)

        #send candidates to slack if len(finalidxs) > 0:
        #make diagnostic plot
        printlog("making diagnostic plot...",output_file=cutterfile,end='')
        canddict = dict()
        canddict['ra_idxs'] = [finalcands[j][0] for j in finalidxs]
        canddict['dec_idxs'] = [finalcands[j][1] for j in finalidxs]
        canddict['wid_idxs'] = [finalcands[j][2] for j in finalidxs]
        canddict['dm_idxs'] = [finalcands[j][3] for j in finalidxs]
        canddict['snrs'] = [finalcands[j][-1] for j in finalidxs]
        canddict['names'] = allcandnames
        if args['classify']:
            canddict['probs'] = probabilities
            canddict['predicts'] = predictions
        if useTOA: 
            canddict['TOAs'] = [finalcands[j][4] for j in finalidxs]
        #RA_axis,DEC_axis,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs)

        # dedisperse to each unique dm candidate
        timeseries = []
        sourceimg_all = np.concatenate([np.zeros(tuple(list(image.shape[:2])+[maxshift]+[image.shape[3]])),image],axis=2)
        for i in range(len(finalidxs)):
            DM = DM_trials[int(canddict['dm_idxs'][i])]

            sourceimg = sourceimg_all[int(canddict['dec_idxs'][i]):int(canddict['dec_idxs'][i])+1,
                                    int(canddict['ra_idxs'][i]):int(canddict['ra_idxs'][i])+1,:,:]#np.concatenate([np.zeros((1,1,maxshift,image.shape[3])),image[canddict['dec_idxs'][i],canddict['ra_idxs'][i],:,:],axis=2)
            printlog("COMPUTING SHIFTS FOR DM="+str(DM)+"pc/cc",output_file=cutterfile)
            freq_axis = np.linspace(fmin,fmax,nchans)
            corr_shifts_all_append,tdelays_frac_append,corr_shifts_all_no_append,tdelays_frac_no_append,wraps_append,wraps_no_append = gen_dm_shifts(np.array([DM]),freq_axis,tsamp,nsamps,outputwraps=True,maxshift=maxshift)

            printlog("corr shifts shape:" + str(corr_shifts_all_append.shape),output_file=cutterfile)

            DM_idx = 0#list(DM_trials).index(DM)
            printlog("PRE-DM SHAPE:"+str(sourceimg.shape),output_file=cutterfile)
            sourceimg_dm = (((((np.take_along_axis(sourceimg[:,:,:,np.newaxis,:].repeat(1,axis=3).repeat(2,axis=4),indices=corr_shifts_all_append[:,:,:,DM_idx:DM_idx+1,:],axis=2))*tdelays_frac_append[:,:,:,DM_idx:DM_idx+1,:]))[:,:,:,0,:]))
            printlog("POST-DM SHAPE:"+str(sourceimg_dm.shape),output_file=cutterfile)
            #zero out anywhere that was wrapped
            #sourceimg_dm[wraps_no_append[:,:,:,DM_idx,:].repeat(sourceimg.shape[0],axis=0).repeat(sourceimg.shape[1],axis=1)] = 0

            #now average the low and high shifts 
            sourceimg_dm = (sourceimg_dm.reshape(tuple(list(sourceimg.shape)[:2] + [nsamps,nchans] + [2])).sum(4))

            timeseries.append(sourceimg_dm.mean((0,1,3)))
        
            if not injection_flag:
                #create json file
                snr=canddict['snrs'][i]
                width=int(widthtrials[int(canddict['wid_idxs'][i])])
                dm=int(DM_trials[int(canddict['dm_idxs'][i])])
                ra=RA_axis_2D[int(canddict['dec_idxs'][i]),int(canddict['ra_idxs'][i])] #RA_axis[int(canddict['ra_idxs'][i])]
                dec=DEC_axis_2D[int(canddict['dec_idxs'][i]),int(canddict['ra_idxs'][i])] #DEC_axis[int(canddict['dec_idxs'][i])]
                trigname = canddict['names'][i]
                printlog(str(snr) +","+ str(width)+","+str(dm) + ","+ str(ra) + "," + str(dec) + "," + trigname,output_file=cutterfile)
                if useTOA:
                    toa = canddict['TOAs'][i]
                    cand_mjd = Time(Time(cand_isot,format='isot').mjd + (canddict['TOAs'][i]*tsamp/1000/86400),format='mjd').mjd
                else:
                    cand_mjd = Time(cand_isot,format='isot').mjd
                
                fl = T4m.nsfrb_to_json(cand_isot,cand_mjd,snr,width,dm,ra,dec,trigname,final_cand_dir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + trigname + "/")
                printlog(fl,output_file=cutterfile)
                if args['trigger']:
                    T4m.submit_cand_nsfrb(fl, logfile=cutterfile)



        candplot=pl.search_plots_new(canddict,image,cand_isot,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                            DM_trials=DM_trials,widthtrials=widthtrials,
                                            output_dir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/",show=False,s100=args['SNRthresh']/2,
                                            injection=injection_flag,vmax=args['SNRthresh']+2,vmin=args['SNRthresh'],
                                            searched_image=searched_image,timeseries=timeseries,uv_diag=uv_diag,dec_obs=dec_obs)
        printlog("done!",output_file=cutterfile)

        if args['toslack']:
            printlog("sending plot to slack...",output_file=cutterfile)
            send_candidate_slack(candplot,filedir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/")
            printlog("done!",output_file=cutterfile)
    
    #cp fast visibilities
    if not injection_flag:
        fastvislabel,fastvisoffset = find_fast_vis_label(cand_mjd)
        printlog("saving candidate visibilities labeled" + str(fastvislabel),output_file=cutterfile)
        printlog("cp " + vis_dir + "lxd110*/*" + str(fastvislabel) + "*.out " + final_cand_dir + "candidates/" + cand_isot + "/",output_file=cutterfile)
        os.system("cp " + vis_dir + "lxd110*/*" + str(fastvislabel) + "*.out " + final_cand_dir + "candidates/" + cand_isot + "/")
    #move fast visibilities, should be labelled with ISOT timestamp
    #if not injection_flag:
    #    os.system("mv " + vis_dir + "lxd110*/*" + cand_isot + "*.out " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/")


    """
    #once finished, move raw data to backup directory if there are remaining candidates, otherwise, delete (at some point, make this an scp to dsastorage)
    if len(finalidxs) > 0:
        os.system("mv " + raw_cand_dir + "*" + cand_isot + "* " + backup_cand_dir)
    else:
        os.system("rm " + raw_cand_dir + "*" + cand_isot + "*")
    """
    #send final candidates to T4 because they will be removed from h24 when it runs out of space
    if args['archive'] and len(finalidxs) > 0 and 'NSFRBT4' in os.environ.keys():
        #make a new directory for timestamp on T4
        T4dir = os.environ['NSFRBT4']
        if injection_flag:
            T4dir += "injections"
        else:
            T4dir += "candidates"
        printlog("ssh user@dsa-storage.ovro.pvt \"mkdir "+ T4dir + "/" + cand_isot+"\"",output_file=cutterfile)
        os.system("ssh user@dsa-storage.ovro.pvt \"mkdir "+ T4dir + "/" + cand_isot+"\"")
        

        #copy csv and cand plot
        printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + cand_isot + "_NSFRBcandplot.png user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/",output_file=cutterfile)
        os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + cand_isot + "_NSFRBcandplot.png user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/")
        printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/"+ "final_candidates_" + cand_isot + ".csv user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/",output_file=cutterfile)
        os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/"+  "final_candidates_" + cand_isot + ".csv user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/")

        #make folder for each candidate
        printlog("ssh user@dsa-storage.ovro.pvt \"mkdir "+ " ".join([T4dir + "/" + cand_isot + "/" + lastname for lastname in allcandnames]) + "\"",output_file=cutterfile)
        os.system("ssh user@dsa-storage.ovro.pvt \"mkdir "+ " ".join([T4dir + "/" + cand_isot + "/" + lastname for lastname in allcandnames]) + "\"")
        printlog("ssh user@dsa-storage.ovro.pvt \"mkdir "+ " ".join([T4dir + "/" + cand_isot + "/" + lastname + "/voltages/" for lastname in allcandnames]) + "\"",output_file=cutterfile)
        os.system("ssh user@dsa-storage.ovro.pvt \"mkdir "+ " ".join([T4dir + "/" + cand_isot + "/" + lastname + "/voltages/" for lastname in allcandnames]) + "\"")
        for lastname in allcandnames:
            #copy numpy files
            printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + lastname + "/*" + " user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/" + lastname + "/",output_file=cutterfile)
            os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/" + lastname + "/*" + " user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/" + lastname + "/")
            
            #once we figure out etcd, also copy voltage files
        #once we figure out etcd, also copy visibility files if offline
        
        #copy fast visibilities
        printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/*.out user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/",output_file=cutterfile)
        os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/*.out user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/")

    printlog("Done! Total Remaining Candidates: " + str(len(finalidxs)),output_file=cutterfile)
    return



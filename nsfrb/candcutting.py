import numpy as np
from nsfrb.config import nsamps as init_nsamps
from nsfrb.config import NSFRB_CANDDADA_KEY,NSFRB_SRCHDADA_KEY,NSFRB_TOADADA_KEY,NSFRB_CANDDADA_SLOW_KEY,NSFRB_SRCHDADA_SLOW_KEY,NSFRB_TOADADA_SLOW_KEY,NSFRB_CANDDADA_IMGDIFF_KEY,NSFRB_SRCHDADA_IMGDIFF_KEY,NSFRB_TOADADA_IMGDIFF_KEY
from realtime.rtreader import rtread_cand
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
from nsfrb.classifying_with_time import classify_images_3D
from nsfrb.noise import init_noise,noise_update_all,get_noise_dict
import hdbscan
import copy
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from nsfrb.config import tsamp,tsamp_slow,fmin,fmax,nchans,nsamps,NUM_CHANNELS, CH0, CH_WIDTH, AVERAGING_FACTOR, IMAGE_SIZE, c, Lon,Lat, DM_tol,table_dir,tsamp_imgdiff
from nsfrb.searching import gen_dm_shifts,widthtrials,DM_trials,DM_trials_slow,gen_boxcar_filter,default_PSF
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
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file,pixperFWHM,candplotfile,candplotfile_slow,candplotfile_imgdiff,candplotupdatefile

from nsfrb.config import freq_axis
#freq_axis = np.linspace(fmin,fmax,nchans)
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
def hdbscan_cluster(cands,min_cluster_size=50,dmt=[0]*16,wt=[0]*5,SNRthresh=1,plot=False,show=False,output_file=cuttertaskfile,PSF=None,min_samples=2,useTOA=False,perc=90,avgcluster=False):
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
def is_injection(isot,inject_file=inject_file,tsamp=tsamp,nsamps=nsamps):
    #check if the candidate is an injection
    injection = False
    postinjection = False
    with open(inject_file,"r") as csvfile:
        re = csv.reader(csvfile,delimiter=',')
        i = 0
        for row in re:
            if i != 0:
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

def sort_cands(fname,image_tesseract_binned,TOAs,SNRthresh,RA_axis,DEC_axis,widthtrials,DM_trials,canddict,raidx_offset=0,decidx_offset=0,output_file=cutterfile,dm_offset=0,maxcands=np.inf,writeraw=False):

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





#main cand cutter task function
import tracemalloc
from nsfrb.config import candcutter_memory_file,candcutter_time_file
def candcutter_task(fname,uv_diag,dec_obs,img_shape,img_search_shape,args):
    """
    Main task to obtain cutouts
    """
    tracemalloc.start()
    s1_ = tracemalloc.take_snapshot()
    t1_ = time.time()
    #for each candidate get the isot and find the corresponding image
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

    try:
        """
        if args['realtime']:
            if slow:
                rtkey1 = NSFRB_CANDDADA_SLOW_KEY
                rtkey2 = NSFRB_SRCHDADA_SLOW_KEY
                rtkey3 = NSFRB_TOADADA_SLOW_KEY
            elif imgdiff:
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
        else:
        """
        image = np.load(raw_cand_dir + cand_isot + suff + ".npy")
        searched_image = np.load(raw_cand_dir + cand_isot + suff + "_searched.npy")
        TOAs = np.load(raw_cand_dir + cand_isot + suff + "_TOAs.npy").astype(int)
    except Exception as e:
        printlog("No image found for candidate " + cand_isot,output_file=cutterfile)
        printlog(str(e),output_file=cutterfile)
        os.system("rm " +  raw_cand_dir + "*" + cand_isot + "*")
        return
    
    t1_end = time.time()-t1_
    s1_end = (tracemalloc.take_snapshot().compare_to(s1_,'lineno'))[0].size_diff

    s2_ = tracemalloc.take_snapshot()
    t2_ = time.time()
    #pad searched image with zeros
    """
    searched_image = np.pad(searched_image,((0,0),
                                            (image.shape[1]-searched_image.shape[1],0),
                                            (0,0),
                                            (0,0)))
    printlog("Padded searched image: " + str(searched_image.shape) + str(image.shape),output_file=cutterfile)
    """
    cand_mjd = Time(cand_isot,format='isot').mjd
    injection_flag,postinjection_flag = is_injection(cand_isot)
    RA_axis,DEC_axis,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs,pixperFWHM=pixperFWHM)
    RA_axis = RA_axis[-searched_image.shape[1]:]
    RA_axis_2D,DEC_axis_2D,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs,two_dim=True,pixperFWHM=pixperFWHM)
    RA_axis_2D = RA_axis_2D[:,-searched_image.shape[1]:]
    DEC_axis_2D = DEC_axis_2D[:,-searched_image.shape[1]:]
    nsamps = image.shape[2]
    canddict = dict()
    raw_cand_names,finalcands = sort_cands(fname,
                    searched_image,TOAs,
                    args['SNRthresh'],
                    RA_axis,DEC_axis,
                    widthtrials,DM_trials_use,canddict,
                    raidx_offset=np.abs(image.shape[1]-searched_image.shape[1]),
                    decidx_offset=np.abs(image.shape[0]-searched_image.shape[0]),
                    maxcands=args['maxcands'],writeraw=args['writeraw'])

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
        
    t2_end = time.time()-t2_
    s2_end = (tracemalloc.take_snapshot().compare_to(s2_,'lineno'))[0].size_diff

    s3_ = tracemalloc.take_snapshot()
    t3_ = time.time()

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
    """try:
        image = np.load(raw_cand_dir + cand_isot + ".npy")
        searched_image = np.load(raw_cand_dir + cand_isot + "_searched.npy")
    except Exception as e:
        printlog("No image found for candidate " + cand_isot,output_file=cutterfile)
        return"""
    """RA_axis,DEC_axis,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs)
    RA_axis = RA_axis[-searched_image.shape[1]:]
    RA_axis_2D,DEC_axis_2D,tmp = uv_to_pix(cand_mjd,image.shape[0],uv_diag=uv_diag,DEC=dec_obs,two_dim=True)
    RA_axis_2D = RA_axis_2D[:,-searched_image.shape[1]:]
    DEC_axis_2D = DEC_axis_2D[:,-searched_image.shape[1]:]
    nsamps = image.shape[2]"""
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
        for i in range(args['clusteriters']):
            mincluster = int(np.max([args['mincluster']//(i+1),2]))

            printlog("Cluster iteration " + str(i+1) + "/" + str(args['clusteriters']) + " with min cluster size " + str(mincluster),output_file=cutterfile)
            if useTOA:
                classes,cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs,centroid_TOAs = hdbscan_cluster(cands_noninf,min_cluster_size=mincluster,min_samples=args['minsamples'],dmt=DM_trials_use,wt=widthtrials,plot=False,show=False,SNRthresh=args['SNRthresh'],PSF=(PSF if i==0 else None),useTOA=True,perc=args['psfpercentile'],avgcluster=args['avgcluster'])
            else:
                classes,cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs = hdbscan_cluster(cands_noninf,min_cluster_size=mincluster,min_samples=args['minsamples'],dmt=DM_trials_use,wt=widthtrials,plot=False,show=False,SNRthresh=args['SNRthresh'],PSF=(PSF if i==0 else None),perc=args['psfpercentile'],avgcluster=args['avgcluster'])
            if np.all(np.array(classes)==-1):
                printlog("Minimum number of clusters reached",output_file=cutterfile)
                break
            else:
                printlog("done, made " + str(len(cluster_cands)) + " clusters",output_file=cutterfile)
                printlog(classes,output_file=cutterfile)
                printlog(cluster_cands,output_file=cutterfile)
             
                cands_noninf = cluster_cands
            
        finalidxs = np.arange(len(cands_noninf),dtype=int)
        finalcands = cands_noninf
        
    #cut by S/N if still too many
    if args['maxcand']:
        printlog("Identifying max S/N candidate",output_file=cutterfile)
        sortedcands = list(np.array(finalcands)[np.argsort(np.array(finalcands)[:,-1])[::-1],:])
        finalcands = sortedcands[0:1]
        finalidxs = np.arange(1)
    elif len(finalcands) >args['maxcands_postcluster']:
        printlog(cand_isot + "has too many candidates to process post-clustering (" + str(len(finalcands)) + ">" + str(args['maxcands_postcluster']) + ") limit...",output_file=cutterfile)
        sortedcands = list(np.array(finalcands)[np.argsort(np.array(finalcands)[:,-1])[::-1],:])
        finalcands = sortedcands[:int(args['maxcands_postcluster'])]
        finalidxs = np.arange(len(finalcands),dtype=int)
        printlog("done, cut to " + str(len(finalcands)) + " candidates",output_file=cutterfile)
    

    t3_end = time.time()-t3_
    s3_end = (tracemalloc.take_snapshot().compare_to(s3_,'lineno'))[0].size_diff

    s4_ = tracemalloc.take_snapshot()
    t4_ = time.time()


    classify_flag = (args['classify'] or args['classify3D'])
    if args['classify']:

        if args['subimgpix'] == image.shape[0]:
            printlog(str("IMGDIFF: " if imgdiff else "") + "Using full image for classification and cutouts;"+str(image.shape),output_file=cutterfile)
            data_array = (img_to_classifier_format(np.repeat(image.mean(2),nchans,axis=2),cand_isot,img_dir)[np.newaxis,:,:,:]).repeat(len(finalcands),axis=0)
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
    
    elif args['classify3D']:
        if args['subimgpix'] == image.shape[0]:
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
            data_array = np.zeros((len(finalcands),args['subimgpix'],args['subimgpix'],image.shape[2],image.shape[3]),dtype=np.float32)
            for j in range(len(finalcands)):
                printlog(finalcands[j],output_file=cutterfile)
                #subimg = quick_snr_fft(get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args['subimgpix'],corr_shift=corr_shifts[:,:,:,int(finalcands[j][3]):int(finalcands[j][3])+1,:],tdelay_frac=tdelays_frac[:,:,:,int(finalcands[j][3]):int(finalcands[j][3])+1,:]),widthtrials[int(finalcands[j][2])])
                #subimg = quick_snr_fft(get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args['subimgpix'],dmidx=int(finalcands[j][3])),widthtrials[int(finalcands[j][2])])

                #don't need to dedisperse(?)
                data_array[j,:,:,:,:] = get_subimage(image,int(finalcands[j][0]),int(finalcands[j][1]),save=False,subimgpix=args['subimgpix'])
                printlog("cand shape:" + str(data_array[j,:,:,:].shape),output_file=cutterfile)

        #run classifier
        printlog("still fine",output_file=cutterfile)
        printlog("Start classifying " + str(data_array.shape),output_file=cutterfile)
        predictions, probabilities = classify_images_3D(data_array, args['model_weights3D'], verbose=args['verbose'])
        printlog(predictions,output_file=cutterfile)
        printlog(probabilities,output_file=cutterfile)

        #only save bursts likely to be real
        #finalidxs = finalidxs[~np.array(predictions,dtype=bool)]
        

    #if set, cut out candidates rejected by the classifier
    if classify_flag and args['classcut']:
        printlog("Classifier rejected " + str(np.sum(predictions)) + "/" + str(len(predictions)) + " candidates",output_file=cutterfile)
        finalcands_new = []
        for i in range(len(finalcands)):
            if predictions[i] == 0:
                finalcands_new.append(finalcands[i])
        finalcands = finalcands_new
        if len(finalcands) == 0:
            printlog("No remaining candidates, done",output_file=cutterfile)
            os.system("rm " +  raw_cand_dir + "*" + cand_isot + "*")
            return
        probabilities = probabilities[predictions==0]
        predictions = predictions[predictions==0]
        finalidxs = np.arange(len(finalcands),dtype=int)

    t4_end = time.time()-t4_
    s4_end = (tracemalloc.take_snapshot().compare_to(s4_,'lineno'))[0].size_diff


    s5_ = tracemalloc.take_snapshot()
    t5_ = time.time()


    #if its an injection write the highest SNR candidate to the injection tracker
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

    #get image cutouts and write to file
    if (args['cutout'] or args['train'] or (args['traininject'] and injection_flag and not postinjection_flag)):
        for j in finalidxs:#range(len(finalidxs)):
            if args['subimgpix'] != image.shape[0]:
                subimg = get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args['subimgpix'])#[:,:,int(finalcands[j][2]),:]
            else:
                subimg = image
            #median subtract
            subimg -= np.nanmedian(subimg,(2,3),keepdims=True)

            lastname = allcandnames[j]

            if (not slow) and (not imgdiff) and (args['train'] or (args['traininject'] and injection_flag and not postinjection_flag)):
                printlog(training_dir+ str("simulated/" if injection_flag else "data/") + cand_isot + "_" + str(j),output_file=cutterfile)
                for i in range(subimg.shape[2]):
                    for k in range(subimg.shape[3]):
                        filepath = training_dir+ str("simulated/" if injection_flag else "data/") + cand_isot + "_" + str(j) + "_subband_avg_{F:.2f}_MHz".format(F=CH0 + CH_WIDTH * k * AVERAGING_FACTOR) + ".png"
                        printlog(filepath,output_file=cutterfile)
                        #if useTOA:
                        #    loc = int(finalcands[j][4])
                        #    wid = widthtrials[int(finalcands[j][2])]
                        #    plt.imsave(filepath, subimg[:,:,int(loc+1-(wid//2)):int(loc+1-(wid//2) + wid),k].mean(2), cmap='gray')
                        #else:
                        #plt.imsave(filepath,subimg[:,:,:,k].mean(2),cmap='gray')
                        plt.imsave(filepath, subimg[:,:,i,k], cmap='gray')
                np.save(training_dir+ str("simulated/" if injection_flag else "data/") + cand_isot + "_" + str(j) + ".npy",subimg)

                f = open(training_dir+ str("simulated/" if injection_flag else "data/") + "labels.csv","a")
                f.write(cand_isot + "_" + str(j) + ",-1,end\n")
                f.close()


            #make folder for each candidate
            os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname)
            os.system("mkdir "+ final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname + "/voltages")
            np.save(final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + lastname + "/" + lastname + ".npy",subimg)
    t5_end = time.time()-t5_
    s5_end = (tracemalloc.take_snapshot().compare_to(s5_,'lineno'))[0].size_diff

    s6_ = tracemalloc.take_snapshot()
    t6_ = time.time()


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
                """
                sourceimg_dm = np.zeros_like(sourceimg)
                for j in range(sourceimg_dm.shape[3]):
                    shiftby = int(np.abs((4.15*DM*((1/(freq_axis[0]*1e-3))**2 - (1/(freq_axis[j]*1e-3))**2))/(tsamp_use)))
                    printlog("freq "+str(freq_axis[j]) + ", shift:" + str(shiftby),output_file=cutterfile)
                    sourceimg_dm[0,0,:,j] = np.pad(sourceimg[0,0,:,j],(0,shiftby))[-sourceimg.shape[2]:]
                printlog("COMPLETED BRUTE FORCE DEDISP",output_file=cutterfile)

                """
                
                
                #freq_axis = np.linspace(fmin,fmax,nchans)
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
        
            
            """
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
                
                fl = T4m.nsfrb_to_json(cand_isot,cand_mjd,snr,width,dm,ra,dec,trigname,final_cand_dir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + trigname + "/",slow=slow,imgdiff=imgdiff)
                printlog(fl,output_file=cutterfile)
                if args['trigger']:
                    T4m.submit_cand_nsfrb(fl, logfile=cutterfile)
            """

        
        printlog("RIGHT BEFORE CANDPLOT",output_file=cutterfile)
        printlog(canddict,output_file=cutterfile)
        printlog(len(RA_axis),output_file=cutterfile)
        printlog(image.shape,output_file=cutterfile)
        candplot=pl.search_plots_new(canddict,image,cand_isot,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                            DM_trials=DM_trials_use,widthtrials=widthtrials,
                                            output_dir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/",show=False,s100=args['SNRthresh']/2,
                                            injection=injection_flag,vmax=args['SNRthresh']*2,vmin=args['SNRthresh'],
                                            searched_image=searched_image,timeseries=timeseries,uv_diag=uv_diag,dec_obs=dec_obs,slow=slow,imgdiff=imgdiff)
        printlog("done!",output_file=cutterfile)

        if args['toslack']:
            #printlog("sending plot to slack...",output_file=cutterfile)
            #send_candidate_slack(candplot,filedir=final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/")
            
            printlog("sending plot to custom webserver 9089...",output_file=cutterfile)
            
            if slow:
                os.system("cp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + candplot + " " + candplotfile_slow)
            elif imgdiff:
                os.system("cp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + candplot + " " + candplotfile_imgdiff)
            else:
                os.system("cp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + candplot + " " + candplotfile)
            """
            #notify
            f = open(frame_dir + "lastcand_srvfile.txt","w")
            f.write(candplot)
            f.close()            
            
            #client_socket = socket.socket()
            #client_socket.connect(("ws://localhost",9087))
            #client_socket.send(candplot.encode())
            #client_socket.close()
            """
            printlog("sending notification via x11...",output_file=cutterfile)
            os.system("cp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/" + candplot + " " + os.environ["NSFRBDIR"] + "/scripts/x11display.png")
            os.system("echo " + str((2/3) if imgdiff else 1) + " > "+ os.environ["NSFRBDIR"] + "/scripts/x11size.txt")
            os.system("echo " + candplot + " > "+ os.environ["NSFRBDIR"] + "/scripts/x11alertmessage.txt") 
            #os.system(os.environ["NSFRBDIR"] + "/scripts/run_x11_display.sh &")
            printlog("done!",output_file=cutterfile)
    t6_end = time.time()-t6_
    s6_end = (tracemalloc.take_snapshot().compare_to(s6_,'lineno'))[0].size_diff
    
    s7_ = tracemalloc.take_snapshot()
    t7_ = time.time()


    #cp fast visibilities
    #if (not args['realtime']) and (not injection_flag):
    if not injection_flag:
        fastvislabel,fastvisoffset = find_fast_vis_label(cand_mjd)
        if fastvislabel != -1:
            printlog("saving candidate visibilities labeled" + str(fastvislabel),output_file=cutterfile)
            printlog("cp " + vis_dir + "lxd110*/*" + str(fastvislabel) + "*.out " + final_cand_dir + "candidates/" + cand_isot + "/",output_file=cutterfile)
            os.system("cp " + vis_dir + "lxd110*/*" + str(fastvislabel) + "*.out " + final_cand_dir + "candidates/" + cand_isot + "/")
    #move fast visibilities, should be labelled with ISOT timestamp
    #if not injection_flag:
    #    os.system("mv " + vis_dir + "lxd110*/*" + cand_isot + "*.out " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + "/")


    #os.system("rm " +  raw_cand_dir + "*" + cand_isot + "*")
    
    #once finished, move raw data to backup directory if there are remaining candidates, otherwise, delete (at some point, make this an scp to dsastorage)
    if (not injection_flag) and (len(finalidxs) > 0):
        os.system("mv " + raw_cand_dir + "*" + cand_isot + "* " + final_cand_dir + "candidates/" + cand_isot + "/")
    else:
        os.system("rm " + raw_cand_dir + "*" + cand_isot + "*")
    
    #send final candidates to T4 because they will be removed from h24 when it runs out of space
    if args['archive'] and len(finalidxs) > 0 and 'NSFRBT4' in os.environ.keys():
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
            
            #once we figure out etcd, also copy voltage files
        #once we figure out etcd, also copy visibility files if offline
        
        #copy fast visibilities
        printlog("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/*.out user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/",output_file=cutterfile)
        os.system("scp " + final_cand_dir + str("injections" if injection_flag else "candidates") + "/" + cand_isot + suff + "/*.out user@dsa-storage.ovro.pvt:" + T4dir + "/" + cand_isot + "/")

    printlog("Done! Total Remaining Candidates: " + str(len(finalidxs)),output_file=cutterfile)
    t7_end = time.time()-t7_
    s7_end = (tracemalloc.take_snapshot().compare_to(s7_,'lineno'))[0].size_diff

    fmem = open(candcutter_memory_file,"a")
    fmem.write(str(s1_end) + " " + str(s2_end) + " " + str(s3_end) + " " + str(s4_end) + " " + str(s5_end) + " "+ str(s6_end)+ " "+ str(s7_end)+"\n")
    fmem.close()

    fmem = open(candcutter_time_file,"a")
    fmem.write(str(t1_end) + " " + str(t2_end) + " " + str(t3_end) + " " + str(t4_end) + " " + str(t5_end) + " " + str(t6_end)+ " "+ str(t7_end)+"\n")
    fmem.close()

    return



import numpy as np
import os
import jax
import socket
import time
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
import candcutting as cc

f = open("../metadata.txt","r")
cwd = f.read()[:-1]
f.close()


import sys
sys.path.append(cwd + "/") #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
import csv
import copy

import sys
sys.path.append(cwd + "/") #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
import csv
import copy

from nsfrb.classifying import classify_images, EnhancedCNN, NumpyImageCubeDataset
from nsfrb.noise import init_noise,noise_update_all,get_noise_dict
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
This file runs the "cand_cutter" service which looks for raw candidate files and post-processes them offline. This
includes clustering, classifying, and cutting out sub-images. It will run in the background.
"""

"""s
Directory for output data
"""
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = cwd + "/src/.pipestatus.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = cwd + "/scripts/script_flags/searchlog_flags.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_flags/searchlog_flags.txt"
output_file = cwd + "-logfiles/run_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
processfile = cwd + "-logfiles/process_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt"
cutterfile = cwd + "-logfiles/candcutter_log.txt"
flagfile = cwd + "/process_server/process_flags.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_flags.txt"
cand_dir = cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
raw_cand_dir = cwd + "-candidates/raw_cands/"
backup_cand_dir = cwd + "-candidates/backup_raw_cands/"
final_cand_dir = cwd + "-candidates/final_cands/"
error_file = cwd + "-logfiles/error_log.txt"

"""
Arguments: data file
"""
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack
from nsfrb.imaging import uv_to_pix


def main():
    #redirect stderr
    sys.stderr = open(error_file,"w")

    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutout', action='store_true', help='Get image cutouts around each candidate')
    parser.add_argument('--subimgpix',type=int,help='Length of image cutouts in pixels, default=11',default=11)
    parser.add_argument('--cluster',action='store_true',help='Enable clustering with HDBSCAN')
    parser.add_argument('--plotclusters',action='store_true',help='Plot intermediate plots from HDBSCAN clustering')
    parser.add_argument('--mincluster',type=int,help='Minimum number of candidates required to be made a separate HDBSCAN cluster,default=5',default=5)
    parser.add_argument('--verbose',action='store_true', help='Enable verbose output')
    parser.add_argument('--classify',action='store_true', help='Classify candidates with a machine learning convolutional neural network')
    parser.add_argument('--model_weights', type=str, help='Path to the model weights file',default=cwd + "/simulations_and_classifications/model_weights.pth")
    
    args = parser.parse_args()
   
    printlog("Starting CandCutter...",output_file=cutterfile)
    #start main loop
    while True:

        #look for candidate files in raw cands dir
        rawfiles = glob.glob(raw_cand_dir + "candidates_*.csv")
        if len(rawfiles) == 0: continue

        #for each candidate get the isot and find the corresponding image
        for i in range(len(rawfiles)):
            fname = rawfiles[i]
            cand_isot = fname[fname.index("candidates_")+11:fname.index(".csv")]
            cand_mjd = Time(cand_isot,format='isot').mjd
            #read cand file
            finalcands = []
            raw_cand_names = []
            with open(fname,"r") as csvfile:
                re = csv.reader(csvfile,delimiter=',')
                for r in re:
                    if 'candname' not in r:
                        finalcands.append(np.array(r[1:],dtype=float))
                        raw_cand_names.append(r[0])
            csvfile.close()
            finalidxs = np.arange(len(finalcands),dtype=int)

            #if getting cutouts, read image
            if args.cutout:
                try:
                    image = np.load(raw_cand_dir + cand_isot + ".npy")
                except Exception as e:
                    printlog("No image found for candidate " + cand_isot,output_file=cutterfile)
                    break
            
            #get DM trials from file
            DMtrials = np.load(cand_dir + "DMtrials.npy")
            widthtrials = np.load(cand_dir + "widthtrials.npy")
            SNRthresh = np.load(cand_dir +"SNRthresh.npy")
            corr_shifts = np.load(cand_dir+"DMcorr_shifts.npy")
            tdelays_frac = np.load(cand_dir+"DMdelays_frac.npy")

            #start clustering
            if args.cluster:
                printlog("clustering with HDBSCAN...",output_file=cutterfile)

                #prune candidates with infinite signal-to-noise for clustering
                cands_noninf = []
                for i in finalcands:
                    if not np.isinf(i[-1]): cands_noninf.append(i)


                #clustering with hdbscan
                classes,cluster_cands,centroid_ras,centroid_decs,centroid_dms,centroid_widths,centroid_snrs = cc.hdbscan_cluster(cands_noninf,min_cluster_size=args.mincluster,dmt=DMtrials,wt=widthtrials,plot=True,show=False,SNRthresh=SNRthresh)
                printlog("done, made " + str(len(cluster_cands)) + " clusters",output_file=cutterfile)
                printlog(classes,output_file=cutterfile)
                printlog(cluster_cands,output_file=cutterfile)

                finalidxs = np.arange(len(cluster_cands),dtype=int)
                finalcands = cluster_cands


            if args.classify:
                #make a binned copy for each candidate
                data_array = np.zeros((len(finalcands),args.subimgpix,args.subimgpix,image.shape[3]),dtype=image.dtype)
                for j in range(len(finalcands)):
                    printlog(finalcands[j],output_file=cutterfile)
                    subimg = cc.quick_snr_fft(cc.get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args.subimgpix,corr_shift=corr_shifts[:,:,:,int(finalcands[j][3]),:],tdelay_frac=tdelays_frac[:,:,:,int(finalcands[j][3]),:]),widthtrials[int(finalcands[j][2])])
                    data_array[j,:,:,:] = subimg[:,:,np.argmin(subimg.sum((0,1,3))),:]
                
                #reformat for classifier
                transposed_array = np.transpose(data_array, (0,3,1,2))#cands x frequencies x RA x DEC
                new_shape = (data_array.shape[0], data_array.shape[3], data_array.shape[1], data_array.shape[2])
                merged_array = transposed_array.reshape(new_shape)
            
                #run classifier
                predictions, probabilities = classify_images(merged_array, args.model_weights, verbose=args.verbose)
                printlog(predictions,output_file=cutterfile)
                printlog(probabilities,output_file=cutterfile)
                
                #only save bursts likely to be real
                #finalidxs = finalidxs[~np.array(predictions,dtype=bool)]




            #write final candidates to csv
            prefix = "NSFRB"
            lastname = None     #once we have etcd, change to 'names.get_lastname()'
            allcandnames = []
            csvfile = open(final_cand_dir + "final_candidates_" + cand_isot + ".csv","w")
            wr = csv.writer(csvfile,delimiter=',')
            if args.classify:
                wr.writerow(["candname","RA index","DEC index","WIDTH index", "DM index", "SNR","PROB"])
            else:
                wr.writerow(["candname","RA index","DEC index","WIDTH index", "DM index", "SNR"])
            sysstdout = sys.stdout
            for j in finalidxs:#range(len(finalidxs)):
                with open(cutterfile,"a") as sys.stdout:
                    lastname = names.increment_name(cand_mjd,lastname=lastname)
                sys.stdout = sysstdout                    
                if args.classify:
                    wr.writerow(np.concatenate([[lastname],np.array(finalcands[j],dtype=int),[probabilities[j]]]))
                else:
                    wr.writerow(np.concatenate([[lastname],np.array(finalcands[j],dtype=int)]))
                allcandnames.append(lastname)
            csvfile.close()

            #get image cutouts and write to file
            if args.cutout:
                for j in finalidxs:#range(len(finalidxs)):
                    subimg = cc.get_subimage(image,finalcands[j][0],finalcands[j][1],save=False,subimgpix=args.subimgpix)#[:,:,int(finalcands[j][2]),:]
                    lastname = allcandnames[j]
                    np.save(final_cand_dir + prefix + lastname + ".npy",subimg)

        


            #once finished, move raw data to backup directory (at some point, make this an scp to dsastorage)
            os.system("mv " + raw_cand_dir + "*" + cand_isot + "* " + backup_cand_dir)
            printlog("Done! Total Remaining Candidates: " + str(len(finalidxs)),output_file=cutterfile)
    
if __name__=="__main__":
    main()
            





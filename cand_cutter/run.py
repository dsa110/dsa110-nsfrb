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
from nsfrb import candcutting as cc

#f = open("../metadata.txt","r")
#cwd = f.read()[:-1]
#f.close()
cwd = os.environ['NSFRBDIR']

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
import os
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = cwd + "/src/.pipestatus.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = cwd + "/scripts/script_flags/searchlog_flags.txt"#"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/script_flags/searchlog_flags.txt"
output_file = cwd + "-logfiles/run_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
processfile = cwd + "-logfiles/process_log.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_log.txt"
cutterfile = cwd + "-logfiles/candcutter_log.txt"
flagfile = cwd + "/process_server/process_flags.txt" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/process_server/process_flags.txt"
cand_dir = os.environ['NSFRBDATA'] + "dsa110-nsfrb-candidates/"#cwd + "-candidates/" #"/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/candidates/"
raw_cand_dir = cand_dir + "raw_cands/"
backup_cand_dir = cand_dir + "backup_raw_cands/"
final_cand_dir = cand_dir + "final_cands/"
error_file = cwd + "-logfiles/candcutter_error_log.txt"

"""
Arguments: data file
"""
from nsfrb.outputlogging import printlog
from nsfrb.outputlogging import send_candidate_slack
from nsfrb.imaging import uv_to_pix
from nsfrb import plotting as pl

"""
Dask scheduler
"""
"""
from dask.distributed import Client,Queue,fire_and_forget
QSETUP = False
if 'DASKPORT' in os.environ.keys():
    try:
        QCLIENT = Client("tcp://127.0.0.1:"+os.environ['DASKPORT'],timeout=1,heartbeat_interval=1000)#get_client()
        QSETUP = True
        QWORKERS = ['cand_cutter_WRKR']
        QQUEUE = Queue("cand_cutter_queue")
    except TimeoutError as exc:
        printlog("Scheduler not started, cannot send to queue",output_file=processfile)
    except OSError as exc:
        printlog("Scheduler not started, cannot send to queue",output_file=processfile)
"""
from multiprocessing import Process, Queue
import dsautils.dsa_store as ds
ETCD = ds.DsaStore()
ETCDKEY = f'/mon/nsfrb/candidates'
QQUEUE = Queue()

def etcd_to_queue(etcd_dict,queue=QQUEUE):
    """
    This is a callback function that takes a candidate from etcd and adds it to the cand cutter queue
    """
    printlog("found etcd candidate:" ,output_file=cutterfile)
    printlog(etcd_dict,output_file=cutterfile)
    printlog("putting in queue",output_file=cutterfile)
    queue.put(etcd_dict['candfile'])
    return


def main(args):
    #redirect stderr
    sys.stderr = open(error_file,"w")
    """
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
    parser.add_argument('--toslack',action='store_true',help='Sends Candidate Summary Plots to Slack')
    parser.add_argument('--sleep',type=float,help='Time in seconds to sleep between successive cand_cutter runs; default=0',default=0)
    parser.add_argument('--runtime',type=float,help='Minimum time in seconds to run before sleep cycle; default=60',default=60)
    
    args = parser.parse_args()
    """
    executor = ThreadPoolExecutor(args.maxProcesses)
    printlog("Starting CandCutter...",output_file=cutterfile)
    #if 'DASKPORT' in os.environ.keys() and QSETUP:
    #    printlog("Restarting Dask client...",output_file=cutterfile)
    #    QCLIENT.restart_workers(QWORKERS)
    if args.etcd:
        printlog("Adding ETCD watch on key "+ETCDKEY,output_file=cutterfile)
        ETCD.add_watch(ETCDKEY, etcd_to_queue)


    #start main loop
    while True:
        #if dask scheduler is setup, look for candidates in the queue
        if args.etcd:#'DASKPORT' in os.environ.keys() and QSETUP:
            printlog("Looking for cands in queue:" + str(QQUEUE),output_file=cutterfile)
            fname = raw_cand_dir + str(QQUEUE.get())
            printlog("Cand Cutter found cand file " + str(fname),output_file=cutterfile)
            future = executor.submit(cc.candcutter_task,fname,vars(args))
            #printlog(future.result(),output_file=cutterfile)
            #fire_and_forget(QCLIENT.submit(cc.candcutter_task,fname,vars(args),workers=QWORKERS))
        else:
            #look for candidate files in raw cands dir
            rawfiles = glob.glob(raw_cand_dir + "candidates_*.csv")
            if len(rawfiles) == 0: continue
            
            

            #for each candidate get the isot and find the corresponding image
            for i in range(len(rawfiles)):
                fname = rawfiles[i]
                printlog("Cand Cutter found cand file " + str(fname),output_file=cutterfile)
                cc.candcutter_task(fname,vars(args))

        if args.sleep > 0:
            printlog("Sleeping for " + str(args.sleep/60) + " minutes",output_file=cutterfile)
            time.sleep(args.sleep)
            #if 'DASKPORT' in os.environ.keys() and QSETUP:
            #    printlog("Restarting Dask client...",output_file=cutterfile)
            #    #QCLIENT.restart_workers(QWORKERS)
    return 0
"""
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
            try:
                image = np.load(raw_cand_dir + cand_isot + ".npy")
            except Exception as e:
                printlog("No image found for candidate " + cand_isot,output_file=cutterfile)
                break
            
            #get DM trials from file
            DMtrials = np.load(cand_dir + "DMtrials.npy")
            widthtrials = np.load(cand_dir + "widthtrials.npy")
            SNRthresh = float(np.load(cand_dir +"SNRthresh.npy"))
            corr_shifts = np.load(cand_dir+"DMcorr_shifts.npy")
            tdelays_frac = np.load(cand_dir+"DMdelays_frac.npy")

            #start clustering
            if args.cluster:
                printlog("clustering with HDBSCAN...",output_file=cutterfile)

                #prune candidates with infinite signal-to-noise for clustering
                cands_noninf = []
                for fcand in finalcands:
                    if not np.isinf(fcand[-1]): cands_noninf.append(fcand)


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
                RA_axis,DEC_axis = uv_to_pix(cand_mjd,image.shape[0],Lat=37.23,Lon=-118.2851)
                candplot=pl.search_plots_new(canddict,image,cand_isot,RA_axis=RA_axis,DEC_axis=DEC_axis,
                                            DM_trials=DMtrials,widthtrials=widthtrials,
                                            output_dir=final_cand_dir,show=False,s100=SNRthresh)
                printlog("done!",output_file=cutterfile)

                if args.toslack:
                    printlog("sending plot to slack...",output_file=cutterfile,end='')
                    send_candidate_slack(candplot,filedir=final_cand_dir)
                    printlog("done!",output_file=cutterfile)


            #once finished, move raw data to backup directory (at some point, make this an scp to dsastorage)
            os.system("mv " + raw_cand_dir + "*" + cand_isot + "* " + backup_cand_dir)
            printlog("Done! Total Remaining Candidates: " + str(len(finalidxs)),output_file=cutterfile)
""" 
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
    parser.add_argument('--model_weights', type=str, help='Path to the model weights file',default=cwd + "/simulations_and_classifications/model_weights.pth")
    parser.add_argument('--toslack',action='store_true',help='Sends Candidate Summary Plots to Slack')
    parser.add_argument('--sleep',type=float,help='Time in seconds to sleep between successive cand_cutter runs; default=0',default=0)
    parser.add_argument('--runtime',type=float,help='Minimum time in seconds to run before sleep cycle; default=60',default=60)
    parser.add_argument('--maxProcesses',type=int,help='Maximum number of threads for thread pool; default=5',default=5)
    parser.add_argument('--archive',action='store_true',help='Archive candidates on dsastorage')
    parser.add_argument('--etcd',action='store_true',help='Enable etcd reading/writing of candidates')
    parser.add_argument('--maxcands',type=int,help='Maximum number of candidates searchable in one iteration. Default is full image, 300x300x5x16=7.2e6',default=int(7.2e6 +1))
    parser.add_argument('--percentile',type=int,help='Percentile above which to take candidates, e.g. if 90, candidates with s/n in 90th percentile will be clustered. Default 0',default=0)
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 10',default=10)
    parser.add_argument('--train',action='store_true',help='Save candidate cutouts to the training set for the ML classifier')
    args = parser.parse_args()
    main(args)
    """
    #run this on the specified worker
    if 'DASKPORT' in os.environ.keys():
        future = QCLIENT.submit(main,args,workers=QWORKER)
        future.result()
    else:
        main(args)
    """     





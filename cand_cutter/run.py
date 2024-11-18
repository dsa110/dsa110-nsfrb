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
"""
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
from nsfrb.config import cwd,cand_dir,frame_dir,psf_dir,img_dir,vis_dir,raw_cand_dir,backup_cand_dir,final_cand_dir,inject_dir,training_dir,noise_dir,imgpath,coordfile,output_file,processfile,timelogfile,cutterfile,pipestatusfile,searchflagsfile,run_file,processfile,cutterfile,cuttertaskfile,flagfile,error_file,inject_file,recover_file,binary_file

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
    parser.add_argument('--percentile',type=float,help='Percentile above which to take candidates, e.g. if 90, candidates with s/n in 90th percentile will be clustered. Default 0',default=0.0)
    parser.add_argument('--SNRthresh',type=float,help='SNR threshold, default = 10',default=10)
    parser.add_argument('--train',action='store_true',help='Save candidate cutouts to the training set for the ML classifier')
    parser.add_argument('--trigger',action='store_true',help='Send T4 trigger to copy visibility buffer and voltages for each candidate event')
    parser.add_argument('--useTOA',action='store_true',help='Include TOAs in clustering algorithm')
    parser.add_argument('--psfcluster',action='store_true',help='PSF-based spatial clustering')
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





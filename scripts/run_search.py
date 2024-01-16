import numpy as np
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

#from gen_dmtrials_copy import gen_dm
import argparse

from scipy.ndimage import convolve
from scipy.signal import convolve2d

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

#import search_lib as sl
import sys
sys.path.append("/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/")
from nsfrb import searching as sl
from nsfrb import pipeline

"""
Directory for output data
"""
output_dir = "./"#"/media/ubuntu/ssd/sherman/NSFRB_search_output/"
pipestatusfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt"
searchflagsfile = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/scripts/searchlog_flags.txt"
output_file = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/tmpoutput/run_log.txt"
"""
Arguments: data file
"""
#time1 = time.time()
#read arguments
#parser = argparse.ArgumentParser()
#parser.add_argument('--fname',type=str,help='File name of 4D numpy array containing image with indices in order (RA,DEC,TIME,FREQ)',default=100)#1000)
#args = parser.parse_args()
#print(args)

from nsfrb.printlog import printlog

def main():
    printlog("Begin run_search.py...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',action='store_true')
    args = parser.parse_args()

    #first check that previous pipe finished
    f = open(pipestatusfile,"r")
    pipestatus = f.read()
    f.close()
    if len(pipestatus) > 0:
        if args.verbose:
            printlog(pipestatus)
        return 1


    #get image input from stdin
    datasize = 2*3276928#209408#6553600#6553600#6553600#6553600#6553600#6553600#6553472#3276928*2#409600
    headersize = 128#128#256#128
    chunksize = 128
    output_shape = -1#(32,32,25,16)
    
    printlog("Reading data from pipe...",end='')
    image_tesseract = pipeline.server_handler(datasize=datasize,headersize=headersize,chunksize=chunksize,output_shape=output_shape)
    printlog("Finished, got data of shape " + str(image_tesseract.shape))
    
    if np.all(image_tesseract == -1):
        if args.verbose:
            printlog("read failed")
        f = open(pipestatusfile,"w")
        f.write(sys.argv[0] + " failed")
        f.close()
        return 1
    #print(image_tesseract)

    #run search
    cands,cluster_cands,image_tesseract_searched = sl.run_search(image_tesseract,SNRthresh=30000)

    #get the image cutouts
    #print(cluster_cands,cluster_cands.shape)

    #first get the unique ra,dec coords to cutout
    unique_cands = [(cluster_cands[i][0],cluster_cands[i][1]) for i in range(len(cluster_cands))]
    unique_cands = list(set(unique_cands))

    unique_cands_dm = [(cluster_cands[i][0],cluster_cands[i][1],cluster_cands[i][2]) for i in range(len(cluster_cands))]
    unique_cands_dm = list(set(unique_cands_dm))
    
    #print(len(unique_cands),len(unique_cands_dm))

    #save image cutouts
    subimgpix = 11
    subimgs_dm = np.zeros((len(unique_cands_dm),subimgpix,subimgpix,image_tesseract.shape[2],image_tesseract.shape[3]),dtype=np.float16)
    subimgs = np.zeros((len(unique_cands),subimgpix,subimgpix,image_tesseract.shape[2],image_tesseract.shape[3]),dtype=np.float16)
    for i in range(len(unique_cands)):
        subimgs[i,:,:,:,:] = sl.get_subimage(image_tesseract,unique_cands[i][0],unique_cands[i][1],save=False,subimgpix=subimgpix)
    for i in range(len(unique_cands_dm)):
        subimgs_dm[i,:,:,:,:] = sl.get_subimage(image_tesseract,unique_cands_dm[i][0],unique_cands_dm[i][1],dm=sl.DM_trials[unique_cands_dm[i][2]],save=False,subimgpix=subimgpix)
    
    if args.verbose:
        printlog(subimgs_dm.shape)
        printlog(subimgs.shape)
    
    #combine full subimage output
    subimgs_all = np.zeros((2,len(unique_cands_dm),subimgpix,subimgpix,image_tesseract.shape[2],image_tesseract.shape[3]),dtype=np.float16)
    subimgs_all[0,:,:,:,:,:] = subimgs
    subimgs_all[1,:,:,:,:,:] = subimgs_dm

    #write length to flag file
    f = open(searchflagsfile,"w")
    f.write("datasize: " + str(len(subimgs_all.tobytes().hex())) + ";")
    f.write("outputshape: " + str(subimgs_all.shape) + ";")
    f.write("size: 16;")
    f.close()

    stat = pipeline.pipeout(subimgs_all)
    if stat == -1:
        if args.verbose:
            printlog("output failed")
        f = open(pipestatusfile,"w")
        f.write(sys.argv[0] + " failed")
        f.close()

    return 0
    
if __name__=="__main__":
    main()

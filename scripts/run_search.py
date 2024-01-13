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

"""
Arguments: data file
"""
#time1 = time.time()
#read arguments
#parser = argparse.ArgumentParser()
#parser.add_argument('--fname',type=str,help='File name of 4D numpy array containing image with indices in order (RA,DEC,TIME,FREQ)',default=100)#1000)
#args = parser.parse_args()
#print(args)


def main():
    #first check that previous pipe finished
    f = open(pipestatusfile,"r")
    pipestatus = f.read()
    f.close()
    if len(pipestatus) > 0:
        print(pipestatus)
        return 1


    #get image input from stdin
    datasize = 2*3276928#209408#6553600#6553600#6553600#6553600#6553600#6553600#6553472#3276928*2#409600
    headersize = 128#128#256#128
    chunksize = 128
    output_shape = -1#(32,32,25,16)

    image_tesseract = pipeline.server_handler(datasize=datasize,headersize=headersize,chunksize=chunksize,output_shape=output_shape,verbose=True)
    if np.all(image_tesseract == -1):
        print("read failed")
        f = open(pipestatusfile,"w")
        f.write(sys.argv[0] + " failed")
        f.close()
        return 1
    print(image_tesseract)

    """#run search
    cands,cluster_cands,image_tesseract_searched = sl.run_search(image_tesseract,SNRthresh=30)

    #convert clustered cands to np array
    cluster_cands_arr = np.zeros((len(cluster_cands),len(cluster_cands[0])))
    for i in range(len(cluster_cands)):
        cluster_cands_arr[i,:] = np.array(cluster_cands[0])

    print(cluster_cands_arr)

    #output as bytes
    #cluster_cands_bytes = cluster_cands_arr.tobytes().hex()
    #print(cluster_cands_bytes)
    stat = pipeline.pipeout(cluster_cands_arr)
    if stat == -1:
        print("output failed")
        f = open(pipestatusfile,"w")
        f.write(sys.argv[0] + " failed")
        f.close()"""
    return 0
if __name__=="__main__":
    main()
"""
sl.search_plots(cands,cluster_cands)

#save image cutouts
for i in range(len(cluster_cands)):
    sl.get_subimage(image_tesseract,cluster_cands[i][0],cluster_cands[i][1],dm=sl.DM_trials[cluster_cands[i][2]],save=True)
    sl.get_subimage(image_tesseract,cluster_cands[i][0],cluster_cands[i][1],save=True)

#cluster
#classes,centroid_raidxs,centroid_decidxs,centroid_dmidxs,centroid_wididxs,centroid_snrs=hdbscan_cluster(cluster_cands,min_cluster_size=100,plot=True)
"""

#print("Total execution time: " + str(time.time()-time1) + " s")

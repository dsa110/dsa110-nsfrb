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

from gen_dmtrials_copy import gen_dm
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
from nsfrb import searching as sl
"""
Directory for output data
"""
output_dir = "/media/ubuntu/ssd/sherman/NSFRB_search_output/"


"""
Arguments: data file
"""
time1 = time.time()
#read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fname',type=str,help='File name of 4D numpy array containing image with indices in order (RA,DEC,TIME,FREQ)',default=100)#1000)
args = parser.parse_args()
print(args)

#read image from file
image_tesseract = np.load(args.fname)

#run search
cands,cluster_cands,image_tesseract_searched = sl.run_search(image_tesseract,SNRthresh=15)
sl.search_plots(cands,cluster_cands)

#save image cutouts
for i in range(len(cluster_cands)):
    sl.get_subimage(image_tesseract,cluster_cands[i][0],cluster_cands[i][1],dm=sl.DM_trials[cluster_cands[i][2]],save=True)
    sl.get_subimage(image_tesseract,cluster_cands[i][0],cluster_cands[i][1],save=True)

#cluster
#classes,centroid_raidxs,centroid_decidxs,centroid_dmidxs,centroid_wididxs,centroid_snrs=hdbscan_cluster(cluster_cands,min_cluster_size=100,plot=True)


print("Total execution time: " + str(time.time()-time1) + " s")

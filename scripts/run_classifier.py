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
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',action='store_true')
    args = parser.parse_args()

    #first check that previous pipe finished
    f = open(pipestatusfile,"r")
    pipestatus = f.read()
    f.close()
    if len(pipestatus) > 0:
        print(pipestatus)
        return 1


    #get data size from flags file
    datasizestr = ''
    while datasizestr == '':
        f = open(searchflagsfile,"r")
        datasizestr = f.read()
        f.close()
    #print(datasizestr)

    dsizeidx = datasizestr.index('datasize:')
    datasize = int(datasizestr[dsizeidx+9:dsizeidx + datasizestr[dsizeidx:].index(";")])
    datasizestr = datasizestr[dsizeidx + datasizestr[dsizeidx:].index(";")+1:]

    dshapeidx = datasizestr.index('outputshape:')
    output_shape = tuple(map(int, datasizestr[dshapeidx+14:dshapeidx + datasizestr[dshapeidx:].index(";")-1].split(',')))
    #output_shape = tuple(datasizestr[dshapeidx+11:dshapeidx + datasizestr[dshapeidx:].index(";")])
    datasizestr = datasizestr[dshapeidx + datasizestr[dshapeidx:].index(";")+1:]

    sizeidx = datasizestr.index('size:')
    bytesize = int(datasizestr[sizeidx+5:sizeidx + datasizestr[sizeidx:].index(";")])#tuple(map(int, datasizestr[dshapeidx+14:dshapeidx + datasizestr[dshapeidx:].index(";")-1].split(',')))
    #output_shape = tuple(datasizestr[dshapeidx+11:dshapeidx + datasizestr[dshapeidx:].index(";")])
    datasizestr = datasizestr[sizeidx + datasizestr[sizeidx:].index(";")+1:]

    #print(datasize,output_shape)  
    #get image input from stdin
    headersize = 0#128
    chunksize = 128
    subimgall = pipeline.server_handler(datasize=datasize,headersize=headersize,chunksize=chunksize,output_shape=output_shape,verbose=args.verbose,bytesize=bytesize)

    subimgs = subimgall[0,:,:,:,:,:]
    subimgs_dm = subimgall[1,:,:,:,:,:]

    print(subimgs,subimgs.shape)
    print(subimgs_dm,subimgs_dm.shape)
    
    #****INSERT CODE FOR ML CLASSIFIER HERE****#
    

    """stat = pipeline.pipeout(cluster_cands_arr)
    if stat == -1:
        print("output failed")
        f = open(pipestatusfile,"w")
        f.write(sys.argv[0] + " failed")
        f.close()"""
    return 0
if __name__=="__main__":
    main()
